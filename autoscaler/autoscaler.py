#!/usr/bin/env python3
"""
GitHub Runner Autoscaler
Manages runner containers based on system resources.

Two loops:
- Fast (2s): pause/unpause based on real-time CPU/memory
- Slow (60s): scaling decisions and stats collection

Priority-based decisions:
- Repos with more recent activity get higher priority
- Repos with shorter test times get higher priority
- High CPU-consuming idle runners are paused first
- Busy runners are never paused
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import docker
import psutil

# Configuration
STATS_FILE = Path(os.environ.get("STATS_FILE", "/data/autoscaler.json"))
FAST_LOOP_INTERVAL = int(os.environ.get("FAST_LOOP_INTERVAL", "2"))
SLOW_LOOP_INTERVAL = int(os.environ.get("SLOW_LOOP_INTERVAL", "60"))

# Resource thresholds (percentage of total)
CPU_HEADROOM_PERCENT = int(os.environ.get("CPU_HEADROOM_PERCENT", "20"))
MEMORY_HEADROOM_PERCENT = int(os.environ.get("MEMORY_HEADROOM_PERCENT", "20"))

# Scaling config
MIN_RUNNERS = int(os.environ.get("MIN_RUNNERS", "1"))  # Always keep at least this many running
IDLE_TIMEOUT_MINUTES = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "10"))
SPAWN_COOLDOWN_SECONDS = int(os.environ.get("SPAWN_COOLDOWN_SECONDS", "30"))  # Min time between spawns per repo

# Resource estimates for spawning
RUNNER_CPU_ESTIMATE = 50  # percent CPU for idle runner
RUNNER_MEMORY_ESTIMATE = 100  # MB for idle runner

# GitHub PAT for spawning new runners
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class RunnerState:
    """Current state of a runner container."""
    container_id: str
    name: str
    repo: str  # owner/repo format
    repo_url: str  # full GitHub URL for spawning
    status: str  # running, paused, exited
    is_busy: bool  # running a job vs idle
    cpu_percent: float
    memory_mb: float


@dataclass
class RepoStats:
    """Historical stats for a repo."""
    frecency_score: float = 0.0
    avg_duration_seconds: float = 300.0  # default 5 min
    peak_concurrent: int = 1
    last_job_time: float = 0.0
    job_count: int = 0
    total_duration: float = 0.0


@dataclass
class AutoscalerState:
    """Persistent state for the autoscaler."""
    repo_stats: dict = field(default_factory=dict)
    last_save: float = 0.0

    def get_repo_stats(self, repo: str) -> RepoStats:
        if repo not in self.repo_stats:
            self.repo_stats[repo] = RepoStats()
        return self.repo_stats[repo]

    def to_dict(self) -> dict:
        return {
            "repo_stats": {
                repo: {
                    "frecency_score": stats.frecency_score,
                    "avg_duration_seconds": stats.avg_duration_seconds,
                    "peak_concurrent": stats.peak_concurrent,
                    "last_job_time": stats.last_job_time,
                    "job_count": stats.job_count,
                    "total_duration": stats.total_duration,
                }
                for repo, stats in self.repo_stats.items()
            },
            "last_save": time.time(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutoscalerState":
        state = cls()
        for repo, stats_data in data.get("repo_stats", {}).items():
            stats = RepoStats(
                frecency_score=stats_data.get("frecency_score", 0.0),
                avg_duration_seconds=stats_data.get("avg_duration_seconds", 300.0),
                peak_concurrent=stats_data.get("peak_concurrent", 1),
                last_job_time=stats_data.get("last_job_time", 0.0),
                job_count=stats_data.get("job_count", 0),
                total_duration=stats_data.get("total_duration", 0.0),
            )
            state.repo_stats[repo] = stats
        return state


class Autoscaler:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.state = AutoscalerState()
        self.load_state()
        # Track last spawn time per repo to prevent rapid spawning
        self.last_spawn_time: Dict[str, float] = {}
        # Track seen job identifiers to avoid re-counting old log lines
        self.seen_jobs: set = set()
        # Track last frecency increment time per repo to debounce parallel runners
        self.last_frecency_time: Dict[str, float] = {}

    def load_state(self):
        """Load persistent state from disk."""
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    data = json.load(f)
                self.state = AutoscalerState.from_dict(data)
                log.info(f"Loaded state from {STATS_FILE}")
            except Exception as e:
                log.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Save persistent state to disk."""
        try:
            STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATS_FILE, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save state: {e}")

    def get_runner_containers(self) -> List[RunnerState]:
        """Get all runner containers with their current state."""
        runners = []

        try:
            containers = self.docker_client.containers.list(all=True)
        except Exception as e:
            log.error(f"Failed to list containers: {e}")
            return runners

        for container in containers:
            name = container.name
            # Match runner containers (but not the manager itself)
            if "runner" not in name.lower() or "manager" in name.lower():
                continue

            # Determine repo from container env
            repo = None
            repo_url = None
            try:
                env = container.attrs.get("Config", {}).get("Env", [])
                for e in env:
                    if e.startswith("REPO_URL="):
                        repo_url = e.split("=", 1)[1]
                        # Extract owner/repo from URL
                        parts = repo_url.rstrip("/").split("/")
                        if len(parts) >= 2:
                            repo = f"{parts[-2]}/{parts[-1]}"
                        break
            except Exception:
                pass

            if not repo or not repo_url:
                continue

            status = container.status

            # Check if busy (running a job) by examining logs
            is_busy = False
            if status == "running":
                try:
                    logs = container.logs(tail=100).decode(errors="ignore")
                    lines = logs.split("\n")
                    last_running_idx = -1
                    last_listening_idx = -1
                    for idx, line in enumerate(lines):
                        if "Running job:" in line:
                            last_running_idx = idx
                        elif "Listening for Jobs" in line:
                            last_listening_idx = idx
                    is_busy = last_running_idx > last_listening_idx
                except Exception:
                    pass

            # Get resource usage
            cpu_percent = 0.0
            memory_mb = 0.0
            if status == "running":
                try:
                    stats = container.stats(stream=False)
                    # Calculate CPU percentage
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                                stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                                   stats["precpu_stats"]["system_cpu_usage"]
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * 100.0 * \
                                      stats["cpu_stats"].get("online_cpus", 1)
                    memory_mb = stats["memory_stats"].get("usage", 0) / (1024 * 1024)
                except Exception:
                    pass

            runners.append(RunnerState(
                container_id=container.id,
                name=name,
                repo=repo,
                repo_url=repo_url,
                status=status,
                is_busy=is_busy,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
            ))

        return runners

    def get_available_resources(self) -> Tuple[float, float]:
        """
        Calculate available CPU and memory for runners.
        Returns (available_cpu_percent, available_memory_mb)
        """
        cpu_count = psutil.cpu_count()
        total_cpu = cpu_count * 100
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)

        current_cpu = psutil.cpu_percent(interval=0.1)
        current_memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        # Keep headroom free
        headroom_cpu = total_cpu * (CPU_HEADROOM_PERCENT / 100)
        headroom_memory = total_memory_mb * (MEMORY_HEADROOM_PERCENT / 100)

        available_cpu = total_cpu - current_cpu - headroom_cpu
        available_memory = total_memory_mb - current_memory_mb - headroom_memory

        return max(0, available_cpu), max(0, available_memory)

    def get_repo_priority(self, repo: str) -> float:
        """
        Calculate priority score for a repo.
        Higher = more important = unpause first, pause last.

        Factors:
        - Frecency: recent activity weighted by time decay
        - Duration: shorter tests = higher priority (faster feedback)
        - Peak usage: repos that use more runners get slight boost
        """
        stats = self.state.get_repo_stats(repo)

        # Decay frecency over time (half-life of 1 hour)
        time_since_last = time.time() - stats.last_job_time
        decay = 0.5 ** (time_since_last / 3600)
        frecency = stats.frecency_score * decay

        # Favor repos with shorter test times (inverse sqrt)
        duration_factor = 1.0 / (1.0 + (stats.avg_duration_seconds / 60) ** 0.5)

        # Peak factor (slightly favor repos that historically use more runners)
        peak_factor = 1.0 + (stats.peak_concurrent - 1) * 0.1

        return frecency * duration_factor * peak_factor

    def can_spawn_runner(self) -> bool:
        """Check if resources allow spawning another runner."""
        if not GITHUB_PAT:
            return False  # Can't spawn without PAT

        available_cpu, available_memory = self.get_available_resources()
        return (available_cpu > RUNNER_CPU_ESTIMATE and
                available_memory > RUNNER_MEMORY_ESTIMATE)

    def spawn_runner(self, repo: str, repo_url: str) -> Optional[str]:
        """
        Spawn an ephemeral runner for a repo.
        Returns container ID if successful, None otherwise.
        """
        if not GITHUB_PAT:
            log.warning("Cannot spawn runner: GITHUB_PAT not set")
            return None

        # Check cooldown
        last_spawn = self.last_spawn_time.get(repo, 0)
        if time.time() - last_spawn < SPAWN_COOLDOWN_SECONDS:
            log.debug(f"Skipping spawn for {repo}: cooldown active")
            return None

        try:
            container_name = f"runner-{repo.replace('/', '-')}-{uuid4().hex[:8]}"
            log.info(f"Spawning ephemeral runner for {repo}: {container_name}")

            container = self.docker_client.containers.run(
                image="github-runner-cypress:latest",
                name=container_name,
                detach=True,
                auto_remove=True,  # Auto-remove when stopped
                environment={
                    "REPO_URL": repo_url,
                    "RUNNER_SCOPE": "repo",
                    "ACCESS_TOKEN": GITHUB_PAT,
                    "RUNNER_NAME_PREFIX": repo.split("/")[-1],
                    "RANDOM_RUNNER_SUFFIX": "true",
                    "LABELS": "self-hosted,linux,x64,nelnet,ephemeral",
                    "RUNNER_WORKDIR": "/tmp/github-runner",
                    "EPHEMERAL": "true",
                    "DISABLE_AUTO_UPDATE": "true",
                },
                volumes={
                    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                },
                extra_hosts={"host.docker.internal": "host-gateway"},
                network="blaha-ci-shared",
                healthcheck={
                    "test": ["CMD-SHELL", "pgrep -f 'Runner.Listener' || exit 1"],
                    "interval": 10000000000,  # 10s in nanoseconds
                    "timeout": 5000000000,    # 5s
                    "retries": 3,
                    "start_period": 30000000000,  # 30s
                },
            )

            self.last_spawn_time[repo] = time.time()
            log.info(f"Spawned runner {container_name} for {repo}")
            return container.id

        except Exception as e:
            log.error(f"Failed to spawn runner for {repo}: {e}")
            return None

    async def fast_loop(self):
        """
        Fast loop (every 2 seconds):
        - Check system resources
        - Pause idle runners if over budget (highest CPU first, lowest priority first)
        - Unpause runners if under budget (highest priority first)
        - Never pause busy runners
        - Always keep MIN_RUNNERS running
        """
        while True:
            try:
                runners = self.get_runner_containers()
                available_cpu, available_memory = self.get_available_resources()

                running_runners = [r for r in runners if r.status == "running"]
                paused_runners = [r for r in runners if r.status == "paused"]

                runner_cpu = sum(r.cpu_percent for r in running_runners)
                runner_memory = sum(r.memory_mb for r in running_runners)

                # Add runner usage back (it's included in "current" usage)
                available_cpu += runner_cpu
                available_memory += runner_memory

                idle_runners = [r for r in running_runners if not r.is_busy]
                busy_runners = [r for r in running_runners if r.is_busy]

                # Sort idle runners: lowest priority first, highest CPU first (pause order)
                idle_runners.sort(key=lambda r: (
                    self.get_repo_priority(r.repo),
                    -r.cpu_percent
                ))

                current_running = len(running_runners)

                # Pause if over budget
                if runner_cpu > available_cpu or runner_memory > available_memory:
                    for runner in idle_runners:
                        if current_running <= MIN_RUNNERS:
                            break
                        if runner_cpu <= available_cpu and runner_memory <= available_memory:
                            break

                        log.info(f"Pausing {runner.name} (CPU: {runner.cpu_percent:.1f}%, Mem: {runner.memory_mb:.0f}MB, Priority: {self.get_repo_priority(runner.repo):.2f})")
                        try:
                            container = self.docker_client.containers.get(runner.container_id)
                            container.pause()
                            runner_cpu -= runner.cpu_percent
                            runner_memory -= runner.memory_mb
                            current_running -= 1
                        except Exception as e:
                            log.error(f"Failed to pause {runner.name}: {e}")

                # Unpause if under budget (with hysteresis)
                elif paused_runners and runner_cpu < available_cpu * 0.8 and runner_memory < available_memory * 0.8:
                    # Sort by priority descending (highest priority = unpause first)
                    paused_runners.sort(key=lambda r: -self.get_repo_priority(r.repo))

                    for runner in paused_runners:
                        est_cpu = 50  # Conservative idle estimate
                        est_memory = 100

                        if runner_cpu + est_cpu > available_cpu * 0.9:
                            break
                        if runner_memory + est_memory > available_memory * 0.9:
                            break

                        log.info(f"Unpausing {runner.name} (Priority: {self.get_repo_priority(runner.repo):.2f})")
                        try:
                            container = self.docker_client.containers.get(runner.container_id)
                            container.unpause()
                            runner_cpu += est_cpu
                            runner_memory += est_memory
                        except Exception as e:
                            log.error(f"Failed to unpause {runner.name}: {e}")

                log.debug(
                    f"Status: CPU {runner_cpu:.0f}/{available_cpu:.0f}%, "
                    f"Mem {runner_memory:.0f}/{available_memory:.0f}MB, "
                    f"Runners: {len(busy_runners)} busy, {len(idle_runners)} idle, {len(paused_runners)} paused"
                )

            except Exception as e:
                log.error(f"Fast loop error: {e}")

            await asyncio.sleep(FAST_LOOP_INTERVAL)

    async def slow_loop(self):
        """
        Slow loop (every 60 seconds):
        - Parse container logs for job events
        - Update frecency and duration stats
        - Track peak concurrent usage
        - Dynamic scaling: spawn marginal runners for busy repos
        - Save state to disk
        """
        while True:
            try:
                runners = self.get_runner_containers()

                # Update stats from logs
                for runner in runners:
                    if runner.status not in ("running", "paused"):
                        continue

                    try:
                        container = self.docker_client.containers.get(runner.container_id)
                        logs = container.logs(tail=500).decode(errors="ignore")

                        stats = self.state.get_repo_stats(runner.repo)

                        for line in logs.split("\n"):
                            if "Running job:" in line:
                                # Create unique job identifier: container + job name
                                job_id = f"{runner.container_id}:{line.strip()}"
                                if job_id not in self.seen_jobs:
                                    self.seen_jobs.add(job_id)
                                    # Parse timestamp from log line: "2025-11-29 18:04:17Z: Running job: test"
                                    job_time = time.time()  # fallback
                                    ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})Z?:", line.strip())
                                    if ts_match:
                                        try:
                                            dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                                            job_time = dt.timestamp()
                                        except ValueError:
                                            pass

                                    # Update last_job_time if this job is more recent
                                    if job_time > stats.last_job_time:
                                        stats.last_job_time = job_time

                                    # Debounce frecency: only count once per 60s per repo
                                    # Uses job timestamp so historical logs are counted correctly
                                    last_freq = self.last_frecency_time.get(runner.repo, 0)
                                    if job_time - last_freq > 60:
                                        stats.frecency_score += 1.0
                                        self.last_frecency_time[runner.repo] = job_time
                                        log.info(f"Frecency +1 for {runner.repo}: {line.strip()}")

                    except Exception as e:
                        log.warning(f"Failed to parse logs for {runner.name}: {e}")

                # Track peak concurrent busy runners per repo
                busy_by_repo: Dict[str, int] = {}
                idle_by_repo: Dict[str, int] = {}
                repo_urls: Dict[str, str] = {}  # Store repo_url for spawning

                for runner in runners:
                    if runner.is_busy:
                        busy_by_repo[runner.repo] = busy_by_repo.get(runner.repo, 0) + 1
                    elif runner.status == "running":
                        idle_by_repo[runner.repo] = idle_by_repo.get(runner.repo, 0) + 1
                    # Track repo URLs for spawning
                    if runner.repo not in repo_urls:
                        repo_urls[runner.repo] = runner.repo_url

                for repo, count in busy_by_repo.items():
                    stats = self.state.get_repo_stats(repo)
                    if count > stats.peak_concurrent:
                        stats.peak_concurrent = count

                # === DYNAMIC SCALING ===
                # Pre-spawn runners based on historical peak usage
                # Goal: maintain peak_concurrent idle runners per repo when resources permit
                repos_needing_runners: List[Tuple[str, int]] = []  # (repo, runners_needed)

                for repo in repo_urls:
                    stats = self.state.get_repo_stats(repo)
                    idle_count = idle_by_repo.get(repo, 0)
                    busy_count = busy_by_repo.get(repo, 0)
                    current_total = idle_count + busy_count

                    # Target: at least peak_concurrent runners, minimum 1
                    target = max(1, stats.peak_concurrent)

                    # How many more do we need to reach target idle?
                    # We want `target` idle runners, currently have `idle_count`
                    runners_needed = target - idle_count

                    if runners_needed > 0:
                        repos_needing_runners.append((repo, runners_needed))

                if repos_needing_runners:
                    # Sort by priority (highest first)
                    repos_needing_runners.sort(key=lambda r: -self.get_repo_priority(r[0]))

                    # Spawn for highest-priority repos until resources exhausted
                    for repo, needed in repos_needing_runners:
                        for _ in range(needed):
                            if self.can_spawn_runner():
                                self.spawn_runner(repo, repo_urls[repo])
                            else:
                                log.info(f"Skipping spawn for {repo}: insufficient resources")
                                break
                        else:
                            continue
                        break  # Resource exhausted, stop spawning

                # === CLEANUP STALE EPHEMERAL RUNNERS ===
                # Remove idle ephemeral runners only if we have more idle than peak_concurrent
                # We want to maintain peak_concurrent idle runners ready to go
                for runner in runners:
                    if not runner.is_busy and runner.name.startswith("runner-"):
                        repo_idle = idle_by_repo.get(runner.repo, 0)
                        stats = self.state.get_repo_stats(runner.repo)
                        target_idle = max(1, stats.peak_concurrent)

                        if repo_idle > target_idle:
                            try:
                                container = self.docker_client.containers.get(runner.container_id)
                                log.info(f"Removing excess idle ephemeral runner: {runner.name} (idle: {repo_idle}, target: {target_idle})")
                                container.stop()
                                # Decrement so we don't remove too many in one pass
                                idle_by_repo[runner.repo] = repo_idle - 1
                            except Exception as e:
                                log.warning(f"Failed to stop {runner.name}: {e}")

                self.save_state()

                log.info(
                    f"Stats: {len(runners)} runners, "
                    f"{sum(1 for r in runners if r.is_busy)} busy, "
                    f"{sum(1 for r in runners if r.status == 'paused')} paused"
                )

            except Exception as e:
                log.error(f"Slow loop error: {e}")

            await asyncio.sleep(SLOW_LOOP_INTERVAL)

    async def run(self):
        """Run both loops concurrently."""
        log.info("Starting autoscaler...")
        log.info(f"Config: CPU headroom {CPU_HEADROOM_PERCENT}%, Memory headroom {MEMORY_HEADROOM_PERCENT}%")
        log.info(f"Min runners: {MIN_RUNNERS}, Stats file: {STATS_FILE}")

        await asyncio.gather(
            self.fast_loop(),
            self.slow_loop(),
        )


async def main():
    autoscaler = Autoscaler()
    await autoscaler.run()


if __name__ == "__main__":
    asyncio.run(main())
