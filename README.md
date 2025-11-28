# Nelnet CI Infrastructure

Private CI/CD infrastructure for self-hosted GitHub Actions runners and reusable workflows.

## Components

### 1. Runner Docker Image (`runner-image/`)
Custom GitHub Actions runner with pre-installed dependencies:
- Cypress browser dependencies (xvfb, GTK, etc.)
- Build essentials

### 2. Runner Stacks (`runners/`)
Docker Compose configurations for self-hosted runners:
- `runners/homepage/` - Runners for nelsonblaha/homepage
- `runners/groovitation/` - Runners for Groovitation/video-games

### 3. Reusable Workflows (`.github/workflows/`)
Callable workflows that projects can use:

#### `python-pytest.yml`
Python testing with pytest and optional coverage.
```yaml
test:
  uses: nelsonblaha/nelnet-ci/.github/workflows/python-pytest.yml@main
  with:
    python_version: '3.11'
    test_path: 'tests/unit'
    coverage_min: 50
    env_vars: '{"ADMIN_PASSWORD": "test", "SESSION_SECRET": "test"}'
```

#### `cypress-e2e.yml`
Cypress E2E testing with app startup.
```yaml
e2e:
  uses: nelsonblaha/nelnet-ci/.github/workflows/cypress-e2e.yml@main
  with:
    spec: 'cypress/e2e/*.cy.js'
    start_command: 'cd app && python -m uvicorn main:app --port 5000'
    base_url: 'http://localhost:5000'
    setup_python: true
```

#### `scala-sbt.yml`
Scala testing with sbt.
```yaml
test:
  uses: nelsonblaha/nelnet-ci/.github/workflows/scala-sbt.yml@main
  with:
    java_version: '11'
    sbt_command: 'test'
```

#### `deploy.yml`
Docker Compose deployment.
```yaml
deploy:
  needs: [test]
  uses: nelsonblaha/nelnet-ci/.github/workflows/deploy.yml@main
  with:
    deploy_path: /home/ben/docker/my-project
  secrets:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Setup

### 1. Build the Runner Image
```bash
cd runner-image
docker build -t github-runner-cypress:latest .
```

### 2. Create GitHub PAT
Create a Personal Access Token with `repo` scope.

### 3. Start Runners
```bash
cd runners/homepage
echo "GITHUB_PAT=your_token" > .env
docker compose up -d
```

## Projects Using This Infrastructure

| Project | Repo | Language | Workflows |
|---------|------|----------|-----------|
| Homepage | nelsonblaha/homepage | Python/FastAPI | pytest, cypress, deploy |
| Priorities | nelsonblaha/priorities | Scala/Play | sbt, cypress, deploy |
| El Paso Automation | nelsonblaha/elpasoautomation | Python | pytest, deploy |
| Video Games | Groovitation/video-games | Scala/Play | sbt, deploy |

## Automatic Propagation

When nelnet-ci is updated, it automatically triggers CI runs in all dependent repos:
1. nelnet-ci tests pass
2. `repository_dispatch` events sent to homepage, priorities, elpasoautomation, video-games
3. Those repos re-run their CI with the latest shared workflows

To receive dispatch events, dependent repos need this trigger in their workflow:
```yaml
on:
  push:
    branches: [main, master]
  repository_dispatch:
    types: [nelnet-ci-updated]
```

Requires a `DISPATCH_PAT` secret with `repo` scope for all target repos.

## Security Notes

- Runners only trigger on `push` to main/master (not on PRs)
- PRs from forks cannot trigger runners (security risk)
- PATs are stored in `.env` files (gitignored)
- `DISPATCH_PAT` needs access to all dependent repos
