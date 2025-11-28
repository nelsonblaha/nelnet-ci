"""Tests for nelnet-ci workflow validation."""
import yaml
import os
from pathlib import Path


def test_all_workflows_valid_yaml():
    """All workflow files should be valid YAML."""
    workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
    for workflow_file in workflows_dir.glob("*.yml"):
        with open(workflow_file) as f:
            data = yaml.safe_load(f)
            assert data is not None, f"{workflow_file.name} is empty"
            assert "on" in data or True, f"{workflow_file.name} missing trigger"


def test_reusable_workflows_have_workflow_call():
    """Reusable workflows should have workflow_call trigger."""
    workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
    reusable = ["python-pytest.yml", "cypress-e2e.yml", "scala-sbt.yml", "deploy.yml"]

    for name in reusable:
        workflow_file = workflows_dir / name
        if workflow_file.exists():
            with open(workflow_file) as f:
                data = yaml.safe_load(f)
                triggers = data.get("on", {})
                assert "workflow_call" in triggers, f"{name} missing workflow_call trigger"


def test_runner_dockerfile_exists():
    """Runner Dockerfile should exist."""
    dockerfile = Path(__file__).parent.parent / "runner-image" / "Dockerfile"
    assert dockerfile.exists(), "runner-image/Dockerfile missing"
