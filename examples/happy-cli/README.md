# Happy-CLI Runner Configuration

Self-hosted GitHub Actions runner for the nelsonblaha/happy repository.

## Features

- Runs CI tests (typecheck, build, tests)
- Auto-deploys to nelnet production on push to main branch
- Ephemeral runner (fresh environment for each job)
- Fork sync status monitoring in nelnet-ci dashboard

## Setup

1. **Create GitHub PAT**
   - Go to https://github.com/settings/tokens
   - Create a token with `repo` scope
   - Copy the token

2. **Configure environment**
   ```bash
   cd /home/ben/src/nelnet-ci/runners/happy-cli
   cp .env.example .env
   # Edit .env and paste your GitHub PAT
   vim .env
   ```

3. **Start the runner**
   ```bash
   docker compose up -d
   ```

4. **Verify it's running**
   ```bash
   docker compose logs -f
   # You should see "Listening for Jobs"
   ```

5. **Add to nelnet-ci dashboard**
   - Go to http://nelnet:8101
   - Add repository: `nelsonblaha/happy`
   - Set upstream: `slopus/happy` (for fork sync status)
   - The dashboard will now show:
     - CI status from GitHub Actions
     - Runner health
     - Fork sync status (shows "X behind" when fork is behind upstream)

## Fork Sync Status

The dashboard includes a special fork sync status feature:

- **In sync**: Shows green "✓ synced" badge
- **Behind upstream**: Shows orange "↻ X behind" clickable badge
  - Clicking the badge takes you to GitHub's compare view
  - You can review changes and sync your fork

## Deployment Process

When you push to main:

1. GitHub Actions triggers workflow
2. Tests run on GitHub-hosted runner
3. If tests pass, deployment job runs on self-hosted runner (nelnet)
4. Deployment steps:
   - Pull latest code to `/home/ben/src/happy/happy-cli`
   - Install dependencies with yarn
   - Build the project
   - Restart happy daemon
   - Health check to verify daemon is running

## Logs

View runner logs:
```bash
cd /home/ben/src/nelnet-ci/runners/happy-cli
docker compose logs -f
```

View daemon logs (after deployment):
```bash
tail -f /var/log/happy-daemon.log
```

## Troubleshooting

**Runner not appearing in GitHub**
- Check GitHub PAT is valid and has `repo` scope
- Verify repository URL is correct in docker-compose.yml
- Check runner logs for errors

**Deployment failing**
- Verify `/home/ben/src/happy` exists on nelnet
- Check runner has access to host Docker socket
- Verify happy daemon can bind to port 3005

**Fork sync not showing**
- Ensure upstream_owner and upstream_repo are set in dashboard
- Check GitHub PAT has access to both fork and upstream repos
- Verify network connectivity to GitHub API
