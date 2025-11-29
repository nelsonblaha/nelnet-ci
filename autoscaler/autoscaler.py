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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

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
    repo: str
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
            try:
                env = container.attrs.get("Config", {}).get("Env", [])
                for e in env:
                    if e.startswith("REPO_URL="):
                        url = e.split("=", 1)[1]
                        # Extract owner/repo from URL
                        parts = url.rstrip("/").split("/")
                        if len(parts) >= 2:
                            repo = f"{parts[-2]}/{parts[-1]}"
                        break
            except Exception:
                pass

            if not repo:
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
                                stats.last_job_time = time.time()
                                stats.frecency_score += 1.0

                    except Exception as e:
                        log.warning(f"Failed to parse logs for {runner.name}: {e}")

                # Track peak concurrent busy runners per repo
                busy_by_repo = {}
                for runner in runners:
                    if runner.is_busy:
                        busy_by_repo[runner.repo] = busy_by_repo.get(runner.repo, 0) + 1

                for repo, count in busy_by_repo.items():
                    stats = self.state.get_repo_stats(repo)
                    if count > stats.peak_concurrent:
                        stats.peak_concurrent = count

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
