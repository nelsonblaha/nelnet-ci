# Self-Hosted GitHub Actions CI Infrastructure

A complete self-hosted CI/CD infrastructure for GitHub Actions with:
- **Custom runner image** with Cypress/browser dependencies pre-installed
- **Resource-aware autoscaler** that pauses idle runners when system resources are needed
- **Web dashboard** showing runner status and active jobs
- **Reusable workflows** for Python, Scala, Cypress, and deployment

## Components

### 1. Runner Image (`runner-image/`)

Custom Docker image based on [myoung34/docker-github-actions-runner](https://github.com/myoung34/docker-github-actions-runner) with:
- Cypress browser dependencies (xvfb, GTK, etc.)
- Build essentials

Build: `cd runner-image && docker build -t github-runner-cypress:latest .`

### 2. Autoscaler (`autoscaler/`)

Monitors system resources and dynamically pauses/unpauses idle runners.

Features:
- Pauses idle runners when CPU/memory is tight
- Priority-based decisions (recent activity, shorter tests get priority)
- Never pauses busy runners (won't break in-progress jobs)
- Configurable headroom (default 20% CPU, 20% memory reserved)

See [autoscaler/DESIGN.md](autoscaler/DESIGN.md) for architecture details.

### 3. Dashboard (`dashboard/`)

Web UI showing:
- Runner containers and their status (idle/busy/paused)
- Active workflow jobs per repo
- Fork sync status for repositories (shows when fork is behind upstream)
- Optional Plex transcode awareness

### 4. Reusable Workflows (`.github/workflows/`)

Callable workflows for common CI patterns:

#### `python-pytest.yml`
```yaml
test:
  uses: YOUR_ORG/YOUR_CI_REPO/.github/workflows/python-pytest.yml@main
  with:
    python_version: '3.11'
    test_path: 'tests/'
```

#### `cypress-e2e.yml`
```yaml
e2e:
  uses: YOUR_ORG/YOUR_CI_REPO/.github/workflows/cypress-e2e.yml@main
  with:
    spec: 'cypress/e2e/*.cy.js'
    start_command: 'python -m uvicorn main:app --port 5000'
    base_url: 'http://localhost:5000'
```

#### `scala-sbt.yml`
```yaml
test:
  uses: YOUR_ORG/YOUR_CI_REPO/.github/workflows/scala-sbt.yml@main
  with:
    java_version: '11'
    sbt_command: 'test'
```

#### `deploy.yml`
```yaml
deploy:
  uses: YOUR_ORG/YOUR_CI_REPO/.github/workflows/deploy.yml@main
  with:
    deploy_path: /path/to/deployment
```

## Quick Start

### 1. Create shared network
```bash
docker network create ci-shared
```

### 2. Build the runner image
```bash
cd runner-image
docker build -t github-runner-cypress:latest .
```

### 3. Start a runner
```bash
# Copy example configs
cp examples/runner-docker-compose.yml runners/myrepo/docker-compose.yml
cp examples/runner.env.example runners/myrepo/.env

# Edit with your repo URL and GitHub PAT
vim runners/myrepo/docker-compose.yml
vim runners/myrepo/.env

# Start
cd runners/myrepo
docker compose up -d
```

### 4. (Optional) Start the autoscaler
The autoscaler runs inside a "manager" container that also serves as a runner.
```bash
cp examples/manager-docker-compose.yml github-runners/docker-compose.yml
# Edit and start...
docker compose up -d
```

### 5. (Optional) Start the dashboard
```bash
cd dashboard
cp ../examples/dashboard.env.example .env
# Edit .env with your GitHub token
docker compose up -d
# Access at http://localhost:8101
```

## Configuration

### Environment Variables

**Runners:**
- `REPO_URL`: GitHub repo URL (e.g., `https://github.com/user/repo`)
- `ACCESS_TOKEN`: GitHub PAT with `repo` scope
- `LABELS`: Comma-separated labels for workflow targeting
- `EPHEMERAL`: Set to `true` for fresh container per job

**Autoscaler:**
- `CPU_HEADROOM_PERCENT`: Reserve this % of CPU (default: 20)
- `MEMORY_HEADROOM_PERCENT`: Reserve this % of memory (default: 20)
- `MIN_RUNNERS`: Always keep at least this many running (default: 1)

**Dashboard:**
- `GITHUB_TOKEN`: PAT for fetching workflow info
- `PLEX_URL` / `PLEX_TOKEN`: Optional Plex integration

### Fork Sync Status

The dashboard can track fork repositories and show when they're behind upstream:

1. When adding a repository in the dashboard, click "+ Upstream" to reveal upstream fields
2. Enter the upstream owner and repo name (e.g., for a fork of `slopus/happy`, enter `slopus` and `happy`)
3. The dashboard will show:
   - **In sync**: Green "✓ synced" badge
   - **Behind**: Orange "↻ X behind" clickable badge that links to GitHub's compare view

This is useful for:
- Monitoring when your fork needs to sync with upstream
- Quick access to GitHub's sync/PR interface
- Tracking multiple forks in one dashboard

## Security Notes

- Runners only trigger on `push` to main/master (not on PRs from forks)
- PATs are stored in `.env` files (gitignored)
- Mount only necessary directories into runner containers
- The autoscaler needs Docker socket access to manage containers

## Testing

Run autoscaler unit tests:
```bash
cd autoscaler
pip install pytest docker psutil
pytest test_autoscaler.py -v
```

## License

MIT
