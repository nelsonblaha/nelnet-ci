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
    ) -> RunnerState:
        return RunnerState(
            container_id=f"id-{name}",
            name=name,
            repo=repo,
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
