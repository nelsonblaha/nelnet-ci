# Autoscaler Design

A resource-aware autoscaler for GitHub Actions self-hosted runners that dynamically pauses and unpauses runner containers based on system load.

## Problem

Self-hosted GitHub runners consume resources even when idle. On a shared server running multiple services (media server, databases, etc.), idle runners waste CPU and memory that could be used by other workloads.

## Solution

An autoscaler that:
- Monitors system CPU and memory usage in real-time
- Pauses idle runner containers when resources are tight
- Unpauses runners when resources become available
- Uses priority scoring to decide which runners to pause/unpause first
- Never pauses busy runners (would break in-progress tests)

## Architecture

### Two-Loop Design

**Fast Loop (2 seconds) - Resource Guardian**
- Get system CPU/memory usage via `psutil`
- Get per-container CPU/memory via `docker stats`
- Calculate resource budget:
  - Reserve: configurable headroom (default 20% CPU, 20% memory)
  - Available: total - current usage - headroom
- Pause/unpause decisions:
  - If over budget: pause idle runners, highest CPU consumers first, lowest priority first
  - If under budget + paused runners exist: unpause, highest priority first
  - Never pause busy runners (mid-job)
- Detect busy/idle state from container logs (`"Running job:"` vs `"Listening for Jobs"`)

**Slow Loop (60 seconds) - Stats Collector**
- Parse container logs for job events
- Update frecency scores (how recently/frequently a repo had jobs)
- Track average job duration per repo
- Track peak concurrent runners per repo
- Persist stats to disk for continuity across restarts

### Priority Scoring

```
priority = frecency * duration_factor * peak_factor

where:
  frecency = score * (0.5 ^ hours_since_last_job)  # 1-hour half-life decay
  duration_factor = 1 / (1 + sqrt(avg_minutes))    # shorter tests = higher priority
  peak_factor = 1 + (peak_concurrent - 1) * 0.1    # slight boost for parallel workloads
```

**Pause order** (first = pause first):
1. Idle runners only (never pause busy)
2. Highest current CPU/memory consumers first
3. Lowest priority repos first

**Unpause order** (first = unpause first):
1. Highest priority repos first

### Data Sources

| Data | Source |
|------|--------|
| Container state | `docker ps` |
| Runner busy/idle | Container log parsing |
| CPU/memory per container | `docker stats` API |
| System resources | `psutil` |
| Job frecency | Container log parsing |
| Job durations | Container log parsing |

### Resource Estimates

| Workload | CPU | RAM |
|----------|-----|-----|
| Idle runner | 5% | 50MB |
| Active runner (build/test) | 100-200% | 500MB-2GB |

### Configuration

Environment variables:
- `CPU_HEADROOM_PERCENT`: Reserve this % of CPU (default: 20)
- `MEMORY_HEADROOM_PERCENT`: Reserve this % of memory (default: 20)
- `MIN_RUNNERS`: Always keep at least this many running (default: 1)
- `FAST_LOOP_INTERVAL`: Seconds between resource checks (default: 2)
- `SLOW_LOOP_INTERVAL`: Seconds between stats collection (default: 60)
- `STATS_FILE`: Path to persist stats (default: `/data/autoscaler.json`)

### Hysteresis

To prevent thrashing (rapid pause/unpause cycles):
- Pause when over budget
- Unpause only when under 80% of budget
- Unpause estimates 50% CPU + 100MB per runner being added

## Deployment

The autoscaler runs inside the `nelnet-ci-manager` container as a background process alongside the GitHub runner. This container needs:
- Docker socket mounted (`/var/run/docker.sock`)
- Volume for stats persistence
- Python 3 with `docker` and `psutil` packages

See `/github-runners/Dockerfile.manager` and `/github-runners/entrypoint-manager.sh` for implementation.

## Testing

Unit tests in `test_autoscaler.py` cover:
- RepoStats dataclass behavior
- AutoscalerState persistence (to_dict/from_dict round-trip)
- Resource budget calculations with headroom
- Priority scoring (frecency decay, duration factor, peak factor)
- Pause decision logic (busy runners protected, CPU-sorted, minimum runners)
- Edge cases (empty runner list, missing REPO_URL, manager container exclusion)

Run with: `pytest test_autoscaler.py -v`

## Dynamic Scaling

The autoscaler dynamically spawns ephemeral runner containers when:
1. A repo has all its runners busy (no idle runners)
2. System resources allow for more runners (CPU/RAM headroom maintained)

### Scaling Logic (in slow loop)

```
For each repo:
  if all runners busy AND no idle runners AND resources available:
    spawn ephemeral runner for this repo
```

Spawning is priority-based:
- Repos are sorted by priority (frecency × duration_factor)
- High-priority repos get marginal runners first when resources are constrained
- Low-priority repos may have to wait

### Ephemeral Runners

Spawned runners are configured with `EPHEMERAL=true`:
- Self-destruct after job completes
- Auto-removed by Docker (`auto_remove=True`)
- No manual cleanup needed

### Spawn Cooldown

To prevent rapid spawning before a runner registers:
- 30-second cooldown per repo between spawns
- Configurable via `SPAWN_COOLDOWN_SECONDS`

### Configuration

Additional environment variables for scaling:
- `GITHUB_PAT`: Required for spawning new runners (same PAT used for registration)
- `SPAWN_COOLDOWN_SECONDS`: Minimum time between spawns per repo (default: 30)
- `RUNNER_CPU_ESTIMATE`: Estimated CPU for resource check (default: 50%)
- `RUNNER_MEMORY_ESTIMATE`: Estimated memory for resource check (default: 100MB)

## Limitations

- **Busy runner protection**: If system resources are exhausted while runners are busy, they cannot be paused. The system will compete for resources until jobs complete.
- **Log-based detection**: Busy/idle detection relies on log parsing. If log format changes in the runner image, detection may break.
- **Seed runners required**: At least one runner per repo must exist in docker-compose.yml for the autoscaler to discover the repo and spawn additional runners.
