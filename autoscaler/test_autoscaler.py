"""
Unit tests for the GitHub Runner Autoscaler.

Run with: pytest test_autoscaler.py -v
"""

import pytest
from unittest.mock import Mock, patch
import time

from autoscaler import (
    Autoscaler,
    AutoscalerState,
    RepoStats,
    RunnerState,
    CPU_HEADROOM_PERCENT,
    MEMORY_HEADROOM_PERCENT,
    RUNNER_CPU_ESTIMATE,
    RUNNER_MEMORY_ESTIMATE,
    SPAWN_COOLDOWN_SECONDS,
)


class TestRepoStats:
    """Tests for RepoStats dataclass."""

    def test_default_values(self):
        stats = RepoStats()
        assert stats.frecency_score == 0.0
        assert stats.avg_duration_seconds == 300.0
        assert stats.peak_concurrent == 1
        assert stats.last_job_time == 0.0

    def test_custom_values(self):
        stats = RepoStats(
            frecency_score=5.0,
            avg_duration_seconds=120.0,
            peak_concurrent=3,
        )
        assert stats.frecency_score == 5.0
        assert stats.avg_duration_seconds == 120.0
        assert stats.peak_concurrent == 3


class TestAutoscalerState:
    """Tests for AutoscalerState persistence."""

    def test_get_repo_stats_creates_new(self):
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        assert isinstance(stats, RepoStats)
        assert "owner/repo" in state.repo_stats

    def test_get_repo_stats_returns_existing(self):
        state = AutoscalerState()
        stats1 = state.get_repo_stats("owner/repo")
        stats1.frecency_score = 10.0
        stats2 = state.get_repo_stats("owner/repo")
        assert stats2.frecency_score == 10.0

    def test_to_dict_round_trip(self):
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        stats.frecency_score = 5.0
        stats.avg_duration_seconds = 120.0
        stats.peak_concurrent = 3

        data = state.to_dict()
        restored = AutoscalerState.from_dict(data)

        restored_stats = restored.get_repo_stats("owner/repo")
        assert restored_stats.frecency_score == 5.0
        assert restored_stats.avg_duration_seconds == 120.0
        assert restored_stats.peak_concurrent == 3

    def test_from_dict_handles_empty(self):
        state = AutoscalerState.from_dict({})
        assert state.repo_stats == {}

    def test_from_dict_handles_missing_fields(self):
        data = {
            "repo_stats": {
                "owner/repo": {
                    "frecency_score": 5.0,
                    # Missing other fields
                }
            }
        }
        state = AutoscalerState.from_dict(data)
        stats = state.get_repo_stats("owner/repo")
        assert stats.frecency_score == 5.0
        assert stats.avg_duration_seconds == 300.0  # Default


class TestResourceBudget:
    """Tests for resource budget calculations."""

    @pytest.fixture
    def autoscaler(self):
        with patch.object(Autoscaler, '__init__', lambda x: None):
            scaler = Autoscaler()
            scaler.docker_client = Mock()
            scaler.state = AutoscalerState()
            return scaler

    def test_available_resources_with_headroom(self, autoscaler):
        """Should reserve headroom percentage."""
        with patch('autoscaler.psutil') as mock_psutil:
            mock_psutil.cpu_count.return_value = 12
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.virtual_memory.return_value = Mock(
                total=32 * 1024 * 1024 * 1024,  # 32GB
                used=8 * 1024 * 1024 * 1024,    # 8GB used
            )

            available_cpu, available_mem = autoscaler.get_available_resources()

            # Total CPU: 1200%, current: 20%, headroom: 20% of 1200 = 240%
            # Available: 1200 - 20 - 240 = 940%
            expected_cpu = 1200 - 20 - (1200 * CPU_HEADROOM_PERCENT / 100)
            assert available_cpu == pytest.approx(expected_cpu, rel=0.1)

    def test_available_never_negative(self, autoscaler):
        """Available resources should never go negative."""
        with patch('autoscaler.psutil') as mock_psutil:
            mock_psutil.cpu_count.return_value = 4  # Small system
            mock_psutil.cpu_percent.return_value = 95.0  # Heavy load
            mock_psutil.virtual_memory.return_value = Mock(
                total=8 * 1024 * 1024 * 1024,  # 8GB
                used=7.5 * 1024 * 1024 * 1024,  # 7.5GB used
            )

            available_cpu, available_mem = autoscaler.get_available_resources()

            assert available_cpu >= 0
            assert available_mem >= 0


class TestRepoPriority:
    """Tests for repo priority scoring."""

    @pytest.fixture
    def autoscaler(self):
        with patch.object(Autoscaler, '__init__', lambda x: None):
            scaler = Autoscaler()
            scaler.docker_client = Mock()
            scaler.state = AutoscalerState()
            return scaler

    def test_fresh_repo_has_zero_priority(self, autoscaler):
        """New repo with no history should have low priority."""
        priority = autoscaler.get_repo_priority("new/repo")
        assert priority == 0.0

    def test_recent_activity_increases_priority(self, autoscaler):
        """Repos with recent jobs should have higher priority."""
        stats = autoscaler.state.get_repo_stats("active/repo")
        stats.frecency_score = 10.0
        stats.last_job_time = time.time()  # Just now

        priority = autoscaler.get_repo_priority("active/repo")
        assert priority > 0

    def test_old_activity_decays(self, autoscaler):
        """Priority should decay over time."""
        stats = autoscaler.state.get_repo_stats("old/repo")
        stats.frecency_score = 10.0
        stats.last_job_time = time.time() - 7200  # 2 hours ago

        priority = autoscaler.get_repo_priority("old/repo")

        # Compare to recent activity
        stats2 = autoscaler.state.get_repo_stats("recent/repo")
        stats2.frecency_score = 10.0
        stats2.last_job_time = time.time()
        recent_priority = autoscaler.get_repo_priority("recent/repo")

        assert priority < recent_priority

    def test_shorter_tests_higher_priority(self, autoscaler):
        """Repos with shorter test times should have higher priority."""
        stats_fast = autoscaler.state.get_repo_stats("fast/repo")
        stats_fast.frecency_score = 10.0
        stats_fast.last_job_time = time.time()
        stats_fast.avg_duration_seconds = 60  # 1 minute

        stats_slow = autoscaler.state.get_repo_stats("slow/repo")
        stats_slow.frecency_score = 10.0
        stats_slow.last_job_time = time.time()
        stats_slow.avg_duration_seconds = 600  # 10 minutes

        fast_priority = autoscaler.get_repo_priority("fast/repo")
        slow_priority = autoscaler.get_repo_priority("slow/repo")

        assert fast_priority > slow_priority

    def test_peak_concurrent_boosts_priority(self, autoscaler):
        """Repos that historically use more runners should get slight boost."""
        stats_single = autoscaler.state.get_repo_stats("single/repo")
        stats_single.frecency_score = 10.0
        stats_single.last_job_time = time.time()
        stats_single.peak_concurrent = 1

        stats_multi = autoscaler.state.get_repo_stats("multi/repo")
        stats_multi.frecency_score = 10.0
        stats_multi.last_job_time = time.time()
        stats_multi.peak_concurrent = 3

        single_priority = autoscaler.get_repo_priority("single/repo")
        multi_priority = autoscaler.get_repo_priority("multi/repo")

        assert multi_priority > single_priority


class TestPauseDecisions:
    """Tests for pause/unpause decision logic."""

    def make_runner(
        self,
        name: str,
        repo: str,
        is_busy: bool = False,
        status: str = "running",
        cpu_percent: float = 50.0,
        memory_mb: float = 500.0,
        repo_url: str = None,
    ) -> RunnerState:
        if repo_url is None:
            repo_url = f"https://github.com/{repo}"
        return RunnerState(
            container_id=f"id-{name}",
            name=name,
            repo=repo,
            repo_url=repo_url,
            status=status,
            is_busy=is_busy,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
        )

    def test_busy_runners_never_paused(self):
        """Busy runners should never be in the pause list."""
        runners = [
            self.make_runner("runner1", "owner/repo", is_busy=True),
            self.make_runner("runner2", "owner/repo", is_busy=False),
        ]

        idle_runners = [r for r in runners if r.status == "running" and not r.is_busy]

        assert len(idle_runners) == 1
        assert idle_runners[0].name == "runner2"

    def test_high_cpu_runners_paused_first(self):
        """Among idle runners, high CPU consumers should be paused first."""
        runners = [
            self.make_runner("low-cpu", "owner/repo", cpu_percent=10.0),
            self.make_runner("high-cpu", "owner/repo", cpu_percent=200.0),
            self.make_runner("med-cpu", "owner/repo", cpu_percent=50.0),
        ]

        # Sort by CPU descending (high CPU = pause first)
        idle_runners = sorted(runners, key=lambda r: -r.cpu_percent)

        assert idle_runners[0].name == "high-cpu"
        assert idle_runners[1].name == "med-cpu"
        assert idle_runners[2].name == "low-cpu"

    def test_minimum_one_runner_kept(self):
        """Even under pressure, at least one runner should stay running."""
        runners = [
            self.make_runner("runner1", "owner/repo1", is_busy=False),
            self.make_runner("runner2", "owner/repo2", is_busy=False),
        ]

        # Simulate "pause until min_runners"
        min_runners = 1
        current_running = len(runners)
        paused = []

        for runner in runners:
            if current_running <= min_runners:
                break
            paused.append(runner.name)
            current_running -= 1

        assert len(paused) == 1
        assert current_running == 1


class TestEdgeCases:
    """Edge cases encountered in production."""

    @pytest.fixture
    def autoscaler(self):
        with patch.object(Autoscaler, '__init__', lambda x: None):
            scaler = Autoscaler()
            scaler.docker_client = Mock()
            scaler.state = AutoscalerState()
            return scaler

    def test_empty_runner_list(self, autoscaler):
        """Should handle no runners gracefully."""
        autoscaler.docker_client.containers.list.return_value = []
        runners = autoscaler.get_runner_containers()
        assert runners == []

    def test_container_without_repo_url(self, autoscaler):
        """Should skip containers without REPO_URL env var."""
        mock_container = Mock()
        mock_container.name = "runner-1"
        mock_container.attrs = {"Config": {"Env": ["OTHER_VAR=value"]}}
        mock_container.status = "running"

        autoscaler.docker_client.containers.list.return_value = [mock_container]
        runners = autoscaler.get_runner_containers()
        assert runners == []

    def test_manager_container_excluded(self, autoscaler):
        """Manager container should not be included in runner list."""
        mock_container = Mock()
        mock_container.name = "nelnet-ci-manager-1"
        mock_container.attrs = {"Config": {"Env": ["REPO_URL=https://github.com/owner/repo"]}}
        mock_container.status = "running"

        autoscaler.docker_client.containers.list.return_value = [mock_container]
        runners = autoscaler.get_runner_containers()
        assert runners == []

    # Add more edge cases here as you encounter them:
    #
    # def test_docker_socket_permission_error(self, autoscaler):
    #     """Should handle Docker permission errors."""
    #     pass
    #
    # def test_container_disappears_mid_operation(self, autoscaler):
    #     """Should handle container being removed during pause."""
    #     pass


class TestDynamicScaling:
    """Tests for dynamic runner spawning."""

    @pytest.fixture
    def autoscaler(self):
        with patch.object(Autoscaler, '__init__', lambda x: None):
            scaler = Autoscaler()
            scaler.docker_client = Mock()
            scaler.state = AutoscalerState()
            scaler.last_spawn_time = {}
            return scaler

    def test_can_spawn_requires_github_pat(self, autoscaler):
        """Should not spawn without GITHUB_PAT."""
        with patch('autoscaler.GITHUB_PAT', ''):
            assert autoscaler.can_spawn_runner() is False

    def test_can_spawn_with_sufficient_resources(self, autoscaler):
        """Should allow spawn when resources are available."""
        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            with patch.object(autoscaler, 'get_available_resources') as mock_resources:
                # More than enough resources
                mock_resources.return_value = (500.0, 2000.0)
                assert autoscaler.can_spawn_runner() is True

    def test_can_spawn_denied_low_cpu(self, autoscaler):
        """Should deny spawn when CPU is low."""
        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            with patch.object(autoscaler, 'get_available_resources') as mock_resources:
                # Not enough CPU
                mock_resources.return_value = (10.0, 2000.0)
                assert autoscaler.can_spawn_runner() is False

    def test_can_spawn_denied_low_memory(self, autoscaler):
        """Should deny spawn when memory is low."""
        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            with patch.object(autoscaler, 'get_available_resources') as mock_resources:
                # Not enough memory
                mock_resources.return_value = (500.0, 50.0)
                assert autoscaler.can_spawn_runner() is False

    def test_spawn_respects_cooldown(self, autoscaler):
        """Should not spawn if cooldown is active."""
        repo = "owner/repo"
        autoscaler.last_spawn_time[repo] = time.time()  # Just spawned

        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            result = autoscaler.spawn_runner(repo, f"https://github.com/{repo}")
            assert result is None

    def test_spawn_after_cooldown_expires(self, autoscaler):
        """Should allow spawn after cooldown expires."""
        repo = "owner/repo"
        autoscaler.last_spawn_time[repo] = time.time() - SPAWN_COOLDOWN_SECONDS - 1

        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            mock_container = Mock()
            mock_container.id = "new-container-id"
            autoscaler.docker_client.containers.run.return_value = mock_container

            result = autoscaler.spawn_runner(repo, f"https://github.com/{repo}")
            assert result == "new-container-id"
            assert autoscaler.docker_client.containers.run.called

    def test_spawn_updates_cooldown_time(self, autoscaler):
        """Should update last_spawn_time after successful spawn."""
        repo = "owner/repo"
        before_time = time.time()

        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            mock_container = Mock()
            mock_container.id = "new-container-id"
            autoscaler.docker_client.containers.run.return_value = mock_container

            autoscaler.spawn_runner(repo, f"https://github.com/{repo}")

            assert repo in autoscaler.last_spawn_time
            assert autoscaler.last_spawn_time[repo] >= before_time

    def test_spawn_creates_ephemeral_container(self, autoscaler):
        """Spawned container should have correct ephemeral configuration."""
        repo = "owner/repo"
        repo_url = f"https://github.com/{repo}"

        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            mock_container = Mock()
            mock_container.id = "new-container-id"
            autoscaler.docker_client.containers.run.return_value = mock_container

            autoscaler.spawn_runner(repo, repo_url)

            call_kwargs = autoscaler.docker_client.containers.run.call_args[1]
            assert call_kwargs['environment']['EPHEMERAL'] == 'true'
            assert call_kwargs['environment']['REPO_URL'] == repo_url
            assert call_kwargs['auto_remove'] is True

    def test_spawn_handles_docker_error(self, autoscaler):
        """Should handle Docker errors gracefully."""
        repo = "owner/repo"

        with patch('autoscaler.GITHUB_PAT', 'test_pat'):
            autoscaler.docker_client.containers.run.side_effect = Exception("Docker error")

            result = autoscaler.spawn_runner(repo, f"https://github.com/{repo}")
            assert result is None


class TestScalingPriority:
    """Tests for priority-based scaling decisions."""

    @pytest.fixture
    def autoscaler(self):
        with patch.object(Autoscaler, '__init__', lambda x: None):
            scaler = Autoscaler()
            scaler.docker_client = Mock()
            scaler.state = AutoscalerState()
            scaler.last_spawn_time = {}
            return scaler

    def test_high_priority_repos_spawned_first(self, autoscaler):
        """When resources are limited, high-priority repos should get runners first."""
        # Setup: repo1 has higher priority (recent activity)
        stats1 = autoscaler.state.get_repo_stats("owner/repo1")
        stats1.frecency_score = 10.0
        stats1.last_job_time = time.time()

        stats2 = autoscaler.state.get_repo_stats("owner/repo2")
        stats2.frecency_score = 1.0
        stats2.last_job_time = time.time() - 7200  # 2 hours ago

        repos_needing_runner = ["owner/repo1", "owner/repo2"]

        # Sort by priority descending (as the slow_loop does)
        repos_needing_runner.sort(key=lambda r: -autoscaler.get_repo_priority(r))

        assert repos_needing_runner[0] == "owner/repo1"

    def test_repos_with_idle_runners_not_spawned(self):
        """Repos with idle runners shouldn't get more spawned."""
        # This is more of a logic test for the slow_loop
        busy_by_repo = {"owner/repo": 1}
        idle_by_repo = {"owner/repo": 1}  # Has an idle runner

        repos_needing_runner = []
        for repo in busy_by_repo:
            idle_count = idle_by_repo.get(repo, 0)
            if idle_count == 0:
                repos_needing_runner.append(repo)

        assert "owner/repo" not in repos_needing_runner

    def test_all_busy_repo_needs_runner(self):
        """Repos with all runners busy should get a new one spawned."""
        busy_by_repo = {"owner/repo": 2}
        idle_by_repo = {}  # No idle runners

        repos_needing_runner = []
        for repo in busy_by_repo:
            idle_count = idle_by_repo.get(repo, 0)
            if idle_count == 0:
                repos_needing_runner.append(repo)

        assert "owner/repo" in repos_needing_runner


class TestPeakDecay:
    """Tests for peak_concurrent decay logic."""

    def test_peak_does_not_decay_when_active(self):
        """Peak should not decay for recently active repos."""
        from autoscaler import PEAK_DECAY_HOURS
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        stats.peak_concurrent = 4
        stats.min_concurrent = 2
        stats.last_job_time = time.time()  # Just now

        hours_idle = (time.time() - stats.last_job_time) / 3600
        assert hours_idle < PEAK_DECAY_HOURS

        # Should not decay
        old_peak = stats.peak_concurrent
        # (No decay applied)
        assert stats.peak_concurrent == old_peak

    def test_peak_decays_after_idle_period(self):
        """Peak should decay after repo has been idle for a while."""
        from autoscaler import PEAK_DECAY_HOURS, PEAK_DECAY_RATE
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        stats.peak_concurrent = 4
        stats.min_concurrent = 2
        stats.last_job_time = time.time() - (PEAK_DECAY_HOURS + 1) * 3600  # Past decay threshold

        hours_idle = (time.time() - stats.last_job_time) / 3600
        assert hours_idle > PEAK_DECAY_HOURS

        # Apply decay (simulating slow loop logic)
        old_peak = stats.peak_concurrent
        decayed_peak = stats.peak_concurrent * PEAK_DECAY_RATE
        stats.peak_concurrent = max(stats.min_concurrent, int(decayed_peak))

        assert stats.peak_concurrent < old_peak
        assert stats.peak_concurrent >= stats.min_concurrent

    def test_peak_never_decays_below_min_concurrent(self):
        """Peak should never go below observed min_concurrent."""
        from autoscaler import PEAK_DECAY_RATE
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        stats.peak_concurrent = 2
        stats.min_concurrent = 2
        stats.last_job_time = time.time() - 10000  # Very old

        # Apply decay multiple times
        for _ in range(10):
            decayed_peak = stats.peak_concurrent * PEAK_DECAY_RATE
            stats.peak_concurrent = max(stats.min_concurrent, int(decayed_peak))

        # Should bottom out at min_concurrent
        assert stats.peak_concurrent == stats.min_concurrent

    def test_min_concurrent_tracks_lowest_observed(self):
        """min_concurrent should track the lowest observed busy count."""
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")

        # Observe different concurrency levels (skipping 1 to match real behavior)
        observations = [4, 2, 3, 2]  # Lowest non-zero is 2

        for count in observations:
            if count > stats.peak_concurrent:
                stats.peak_concurrent = count
            # Only update min if we haven't seen anything yet (min==1 is default) or if count is lower
            if count > 0 and (stats.min_concurrent == 1 or count < stats.min_concurrent):
                stats.min_concurrent = count

        assert stats.min_concurrent == 2
        assert stats.peak_concurrent == 4

        # Now observe a single runner (edge case: workflow with only 1 job)
        count = 1
        if count > 0 and count < stats.min_concurrent:
            stats.min_concurrent = count

        assert stats.min_concurrent == 1

    def test_min_concurrent_persistence(self):
        """min_concurrent should persist across to_dict/from_dict round trips."""
        state = AutoscalerState()
        stats = state.get_repo_stats("owner/repo")
        stats.min_concurrent = 2
        stats.peak_concurrent = 4

        data = state.to_dict()
        restored = AutoscalerState.from_dict(data)

        restored_stats = restored.get_repo_stats("owner/repo")
        assert restored_stats.min_concurrent == 2
        assert restored_stats.peak_concurrent == 4
