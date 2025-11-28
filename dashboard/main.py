"""
Nelnet CI Dashboard
- Monitors and manages self-hosted GitHub Actions runners
- Auto-scales runners based on system load and Plex activity
- Manages approved committers and repo configurations
"""

import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import aiosqlite
import docker
import httpx
import psutil
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Configuration
DATABASE_PATH = os.environ.get("DATABASE_PATH", "/data/nelnet-ci.db")
PLEX_URL = os.environ.get("PLEX_URL", "http://plex:32400")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Auto-scaling config (can be updated via API)
DEFAULT_SCALING_CONFIG = {
    "min_runners": 2,
    "max_runners": 6,
    "cpu_reserve_percent": 25,  # Reserve this % of CPU for other services
    "plex_transcode_reserve": 2,  # Reserve capacity for this many transcodes
    "scale_up_threshold": 0.7,  # Scale up when queue > this % of runners
    "scale_down_threshold": 0.2,  # Scale down when queue < this %
    "check_interval_seconds": 30,
}


# Database setup
async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL,
                name TEXT NOT NULL,
                runner_group TEXT DEFAULT 'default',
                allow_pr_tests BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(owner, name)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS approved_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                github_username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        await db.commit()


@asynccontextmanager
async def get_db():
    db = await aiosqlite.connect(DATABASE_PATH)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


# Plex monitoring
async def get_plex_sessions() -> dict:
    """Get current Plex streaming sessions."""
    if not PLEX_TOKEN:
        return {"sessions": 0, "transcodes": 0, "error": "No Plex token configured"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{PLEX_URL}/status/sessions",
                params={"X-Plex-Token": PLEX_TOKEN},
                headers={"Accept": "application/json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("MediaContainer", {}).get("Metadata", [])
                transcode_count = sum(
                    1 for s in sessions
                    if s.get("TranscodeSession") is not None
                )
                return {
                    "sessions": len(sessions),
                    "transcodes": transcode_count,
                    "details": [
                        {
                            "title": s.get("title", "Unknown"),
                            "user": s.get("User", {}).get("title", "Unknown"),
                            "transcoding": s.get("TranscodeSession") is not None,
                        }
                        for s in sessions
                    ]
                }
            return {"sessions": 0, "transcodes": 0, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"sessions": 0, "transcodes": 0, "error": str(e)}


# System monitoring
def get_system_stats() -> dict:
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": cpu_percent,
        "memory_total_gb": round(memory.total / (1024**3), 1),
        "memory_used_gb": round(memory.used / (1024**3), 1),
        "memory_percent": memory.percent,
        "load_avg": list(os.getloadavg()),
    }


# Docker/Runner management
def get_docker_client():
    return docker.from_env()


def get_runner_containers() -> list:
    """Get all GitHub runner containers with job info."""
    client = get_docker_client()
    containers = client.containers.list(all=True)
    runners = []
    for c in containers:
        if "runner" in c.name.lower() and "github" in c.name.lower():
            runner_info = {
                "id": c.short_id,
                "name": c.name,
                "status": c.status,
                "health": c.attrs.get("State", {}).get("Health", {}).get("Status", "unknown"),
                "job": None,
                "repo": None,
            }

            # Try to get job info from runner logs
            if c.status == "running":
                try:
                    # Get repo from .runner config
                    result = c.exec_run("cat /actions-runner/.runner", stderr=False)
                    if result.exit_code == 0:
                        import json
                        runner_config = json.loads(result.output.decode())
                        github_url = runner_config.get("gitHubUrl", "")
                        if github_url:
                            # Extract owner/repo from URL
                            parts = github_url.rstrip("/").split("/")
                            if len(parts) >= 2:
                                runner_info["repo"] = f"{parts[-2]}/{parts[-1]}"

                    # Check latest log for job status
                    result = c.exec_run("sh -c 'cat $(ls -t /actions-runner/_diag/Runner_*.log 2>/dev/null | head -1) | tail -20'", stderr=False)
                    if result.exit_code == 0:
                        log_content = result.output.decode()
                        if "Running job:" in log_content:
                            # Extract job name
                            for line in log_content.split('\n'):
                                if "Running job:" in line:
                                    runner_info["job"] = line.split("Running job:")[-1].strip()
                                    break
                        elif "Listening for Jobs" in log_content:
                            runner_info["job"] = "idle"
                except Exception:
                    pass

            runners.append(runner_info)
    return runners


# GitHub API
async def get_workflow_status(owner: str, repo: str) -> dict:
    """Get latest workflow run status for a repo."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for branch in ["main", "master"]:
                resp = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/actions/runs",
                    params={"per_page": 1, "branch": branch},
                    headers=headers
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("workflow_runs"):
                        run = data["workflow_runs"][0]
                        return {
                            "status": run.get("status"),
                            "conclusion": run.get("conclusion"),
                            "url": run.get("html_url"),
                            "created_at": run.get("created_at"),
                            "branch": branch,
                        }
            return {"status": "no_runs", "conclusion": None}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_user_approved(username: str) -> bool:
    """Check if a GitHub user is in the approved list."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT 1 FROM approved_users WHERE github_username = ?",
            (username.lower(),)
        )
        return await cursor.fetchone() is not None


# API Models
class RepoConfig(BaseModel):
    owner: str
    name: str
    runner_group: str = "default"
    allow_pr_tests: bool = False


class ApprovedUser(BaseModel):
    github_username: str


class ScalingConfig(BaseModel):
    min_runners: int = 2
    max_runners: int = 6
    cpu_reserve_percent: int = 25
    plex_transcode_reserve: int = 2


# Auth
async def verify_admin(request: Request):
    if not ADMIN_PASSWORD:
        return True  # No password configured, allow access
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if token == ADMIN_PASSWORD:
            return True
    raise HTTPException(status_code=401, detail="Unauthorized")


# App setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Nelnet CI Dashboard", lifespan=lifespan)


# API Routes
@app.get("/api/status")
async def get_status():
    """Get overall system and CI status."""
    plex = await get_plex_sessions()
    system = get_system_stats()
    runners = get_runner_containers()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": system,
        "plex": plex,
        "runners": {
            "count": len(runners),
            "healthy": sum(1 for r in runners if r["health"] == "healthy"),
            "containers": runners,
        },
    }


@app.get("/api/repos")
async def list_repos(_: bool = Depends(verify_admin)):
    """List all configured repositories."""
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM repos ORDER BY owner, name")
        rows = await cursor.fetchall()
        repos = [dict(row) for row in rows]

    # Fetch CI status for each repo
    for repo in repos:
        status = await get_workflow_status(repo["owner"], repo["name"])
        repo["ci_status"] = status

    return repos


@app.post("/api/repos")
async def add_repo(repo: RepoConfig, _: bool = Depends(verify_admin)):
    """Add a repository configuration."""
    async with get_db() as db:
        try:
            await db.execute(
                "INSERT INTO repos (owner, name, runner_group, allow_pr_tests) VALUES (?, ?, ?, ?)",
                (repo.owner, repo.name, repo.runner_group, repo.allow_pr_tests)
            )
            await db.commit()
            return {"status": "ok", "message": f"Added {repo.owner}/{repo.name}"}
        except aiosqlite.IntegrityError:
            raise HTTPException(status_code=400, detail="Repository already exists")


@app.delete("/api/repos/{owner}/{name}")
async def remove_repo(owner: str, name: str, _: bool = Depends(verify_admin)):
    """Remove a repository configuration."""
    async with get_db() as db:
        await db.execute("DELETE FROM repos WHERE owner = ? AND name = ?", (owner, name))
        await db.commit()
    return {"status": "ok"}


@app.get("/api/approved-users")
async def list_approved_users(_: bool = Depends(verify_admin)):
    """List approved GitHub users."""
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM approved_users ORDER BY github_username")
        return [dict(row) for row in await cursor.fetchall()]


@app.post("/api/approved-users")
async def add_approved_user(user: ApprovedUser, _: bool = Depends(verify_admin)):
    """Add an approved GitHub user."""
    async with get_db() as db:
        try:
            await db.execute(
                "INSERT INTO approved_users (github_username) VALUES (?)",
                (user.github_username.lower(),)
            )
            await db.commit()
            return {"status": "ok", "message": f"Added {user.github_username}"}
        except aiosqlite.IntegrityError:
            raise HTTPException(status_code=400, detail="User already exists")


@app.delete("/api/approved-users/{username}")
async def remove_approved_user(username: str, _: bool = Depends(verify_admin)):
    """Remove an approved GitHub user."""
    async with get_db() as db:
        await db.execute("DELETE FROM approved_users WHERE github_username = ?", (username.lower(),))
        await db.commit()
    return {"status": "ok"}


@app.get("/api/check-user/{username}")
async def check_user(username: str):
    """Check if a user is approved (for workflow use)."""
    approved = await check_user_approved(username)
    return {"username": username, "approved": approved}


@app.get("/api/config")
async def get_config(_: bool = Depends(verify_admin)):
    """Get current configuration."""
    async with get_db() as db:
        cursor = await db.execute("SELECT key, value FROM config")
        rows = await cursor.fetchall()
        config = {row["key"]: row["value"] for row in rows}

    return {
        "github_token_configured": bool(GITHUB_TOKEN),
        "github_token_preview": GITHUB_TOKEN[:10] + "..." if GITHUB_TOKEN else None,
        "plex_token_configured": bool(PLEX_TOKEN),
        "admin_password_configured": bool(ADMIN_PASSWORD),
        "scaling": config,
    }


@app.get("/api/repos/{owner}/{repo}/badge")
async def get_repo_badge(owner: str, repo: str):
    """Get badge info for a repo - shields.io compatible endpoint."""
    status = await get_workflow_status(owner, repo)

    if status.get("conclusion") == "success":
        color = "brightgreen"
        label = "passing"
    elif status.get("conclusion") == "failure":
        color = "red"
        label = "failing"
    elif status.get("status") == "in_progress":
        color = "yellow"
        label = "running"
    else:
        color = "lightgrey"
        label = "unknown"

    # shields.io only accepts specific properties - url is not allowed
    return {
        "schemaVersion": 1,
        "label": "CI",
        "message": label,
        "color": color,
    }


@app.post("/api/config/{key}")
async def set_config(key: str, value: str, _: bool = Depends(verify_admin)):
    """Set a configuration value."""
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, value)
        )
        await db.commit()
    return {"status": "ok"}


# Simple HTML UI
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Simple dashboard UI."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Nelnet CI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div x-data="dashboard()" x-init="init()" class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8">Nelnet CI Dashboard</h1>

        <!-- System Status -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-2">System</h2>
                <div x-show="status.system">
                    <!-- CPU Progress Bar -->
                    <div class="mb-3">
                        <div class="flex justify-between text-sm mb-1">
                            <span>CPU</span>
                            <span x-text="status.system?.cpu_percent + '%'"></span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-3">
                            <div class="h-3 rounded-full transition-all duration-300"
                                 :class="status.system?.cpu_percent > 80 ? 'bg-red-500' : status.system?.cpu_percent > 50 ? 'bg-yellow-500' : 'bg-green-500'"
                                 :style="'width: ' + status.system?.cpu_percent + '%'"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-1"><span x-text="status.system?.cpu_count"></span> cores</p>
                    </div>
                    <!-- Memory Progress Bar -->
                    <div class="mb-3">
                        <div class="flex justify-between text-sm mb-1">
                            <span>Memory</span>
                            <span x-text="status.system?.memory_used_gb + '/' + status.system?.memory_total_gb + ' GB'"></span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-3">
                            <div class="h-3 rounded-full transition-all duration-300"
                                 :class="status.system?.memory_percent > 80 ? 'bg-red-500' : status.system?.memory_percent > 50 ? 'bg-yellow-500' : 'bg-green-500'"
                                 :style="'width: ' + status.system?.memory_percent + '%'"></div>
                        </div>
                    </div>
                    <p class="text-xs text-gray-500">Load: <span x-text="status.system?.load_avg?.map(l => l.toFixed(2)).join(', ')"></span></p>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-2">Plex</h2>
                <div x-show="status.plex">
                    <p>Sessions: <span x-text="status.plex?.sessions"></span></p>
                    <p>Transcodes: <span x-text="status.plex?.transcodes"></span></p>
                    <p x-show="status.plex?.error" class="text-red-400" x-text="status.plex?.error"></p>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-2">Runners</h2>
                <div x-show="status.runners">
                    <p class="mb-2">Active: <span x-text="status.runners?.healthy + '/' + status.runners?.count"></span></p>
                    <div class="space-y-2">
                        <template x-for="r in status.runners?.containers" :key="r.id">
                            <div class="bg-gray-700/50 rounded p-2">
                                <div class="flex items-center gap-2">
                                    <span class="inline-block w-2 h-2 rounded-full"
                                          :class="r.health === 'healthy' ? 'bg-green-400' : 'bg-yellow-400'"></span>
                                    <span class="text-sm font-medium" x-text="r.name.replace('github-runners-', '')"></span>
                                </div>
                                <div class="text-xs text-gray-400 mt-1 ml-4">
                                    <span x-show="r.repo" x-text="r.repo"></span>
                                    <span x-show="r.job && r.job !== 'idle'" class="text-yellow-400">
                                        → <span x-text="r.job"></span>
                                    </span>
                                    <span x-show="r.job === 'idle'" class="text-gray-500">idle</span>
                                    <span x-show="!r.job && r.status !== 'running'" class="text-gray-500" x-text="r.status"></span>
                                </div>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
        </div>

        <!-- Repos -->
        <div class="bg-gray-800 rounded-lg p-4 mb-8">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-lg font-semibold">Configured Repositories</h2>
                <span class="text-xs text-gray-500">Auto-refreshes every minute</span>
            </div>
            <table class="w-full">
                <thead>
                    <tr class="text-left border-b border-gray-700">
                        <th class="pb-2">Repository</th>
                        <th class="pb-2">CI Status</th>
                        <th class="pb-2">Badge</th>
                        <th class="pb-2">PR Tests</th>
                        <th class="pb-2">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <template x-for="repo in repos" :key="repo.id">
                        <tr class="border-b border-gray-700">
                            <td class="py-2">
                                <a :href="'https://github.com/' + repo.owner + '/' + repo.name"
                                   target="_blank" class="text-blue-400 hover:underline"
                                   x-text="repo.owner + '/' + repo.name"></a>
                            </td>
                            <td class="py-2">
                                <a :href="repo.ci_status?.url" target="_blank" class="flex items-center gap-2">
                                    <span class="inline-block w-3 h-3 rounded-full"
                                          :class="getStatusDotClass(repo.ci_status)"></span>
                                    <span :class="getStatusClass(repo.ci_status)"
                                          x-text="getStatusText(repo.ci_status)"></span>
                                </a>
                            </td>
                            <td class="py-2">
                                <img :src="'https://img.shields.io/endpoint?url=' + encodeURIComponent(window.location.origin + '/api/repos/' + repo.owner + '/' + repo.name + '/badge')"
                                     alt="CI Badge" class="h-5">
                            </td>
                            <td class="py-2">
                                <span :class="repo.allow_pr_tests ? 'text-yellow-400' : 'text-green-400'"
                                      x-text="repo.allow_pr_tests ? 'Enabled (risky!)' : 'Disabled'"></span>
                            </td>
                            <td class="py-2">
                                <button @click="removeRepo(repo)" class="text-red-400 hover:text-red-300">Remove</button>
                            </td>
                        </tr>
                    </template>
                </tbody>
            </table>

            <!-- Add repo form -->
            <div class="mt-4 flex gap-2">
                <input x-model="newRepo.owner" placeholder="owner" class="bg-gray-700 px-2 py-1 rounded">
                <input x-model="newRepo.name" placeholder="repo" class="bg-gray-700 px-2 py-1 rounded">
                <label class="flex items-center gap-1">
                    <input type="checkbox" x-model="newRepo.allow_pr_tests">
                    <span class="text-sm">Allow PR tests</span>
                </label>
                <button @click="addRepo()" class="bg-blue-600 px-3 py-1 rounded hover:bg-blue-500">Add</button>
            </div>
        </div>

        <!-- Configuration -->
        <div class="bg-gray-800 rounded-lg p-4 mb-8">
            <h2 class="text-lg font-semibold mb-4">Configuration</h2>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p class="text-sm text-gray-400">GitHub Token</p>
                    <p :class="config.github_token_configured ? 'text-green-400' : 'text-red-400'"
                       x-text="config.github_token_configured ? 'Configured (' + config.github_token_preview + ')' : 'Not configured'"></p>
                </div>
                <div>
                    <p class="text-sm text-gray-400">Plex Token</p>
                    <p :class="config.plex_token_configured ? 'text-green-400' : 'text-red-400'"
                       x-text="config.plex_token_configured ? 'Configured' : 'Not configured'"></p>
                </div>
            </div>
            <p class="text-sm text-gray-500 mt-4">Tokens are set via environment variables in docker-compose.yml</p>
        </div>

        <!-- Approved Users -->
        <div class="bg-gray-800 rounded-lg p-4">
            <h2 class="text-lg font-semibold mb-4">Approved Committers</h2>
            <div class="flex flex-wrap gap-2 mb-4">
                <template x-for="user in approvedUsers" :key="user.id">
                    <span class="bg-gray-700 px-2 py-1 rounded flex items-center gap-2">
                        <span x-text="user.github_username"></span>
                        <button @click="removeUser(user)" class="text-red-400 hover:text-red-300">&times;</button>
                    </span>
                </template>
            </div>
            <div class="flex gap-2">
                <input x-model="newUsername" placeholder="GitHub username" class="bg-gray-700 px-2 py-1 rounded">
                <button @click="addUser()" class="bg-blue-600 px-3 py-1 rounded hover:bg-blue-500">Add User</button>
            </div>
        </div>
    </div>

    <script>
    function dashboard() {
        return {
            status: {},
            repos: [],
            approvedUsers: [],
            config: {},
            newRepo: { owner: '', name: '', allow_pr_tests: false },
            newUsername: '',
            authHeader: { 'Authorization': 'Bearer ' + (localStorage.getItem('adminToken') || '') },

            async init() {
                await this.refresh();
                setInterval(() => this.loadStatus(), 10000);
                setInterval(() => this.loadRepos(), 60000);  // Refresh repos every minute
            },

            async refresh() {
                await Promise.all([this.loadStatus(), this.loadRepos(), this.loadUsers(), this.loadConfig()]);
            },

            async loadStatus() {
                const resp = await fetch('/api/status');
                this.status = await resp.json();
            },

            async loadRepos() {
                const resp = await fetch('/api/repos', { headers: this.authHeader });
                if (resp.ok) this.repos = await resp.json();
            },

            async loadUsers() {
                const resp = await fetch('/api/approved-users', { headers: this.authHeader });
                if (resp.ok) this.approvedUsers = await resp.json();
            },

            async loadConfig() {
                const resp = await fetch('/api/config', { headers: this.authHeader });
                if (resp.ok) this.config = await resp.json();
            },

            async addRepo() {
                await fetch('/api/repos', {
                    method: 'POST',
                    headers: { ...this.authHeader, 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.newRepo)
                });
                this.newRepo = { owner: '', name: '', allow_pr_tests: false };
                await this.loadRepos();
            },

            async removeRepo(repo) {
                if (!confirm(`Remove ${repo.owner}/${repo.name}?`)) return;
                await fetch(`/api/repos/${repo.owner}/${repo.name}`, {
                    method: 'DELETE',
                    headers: this.authHeader
                });
                await this.loadRepos();
            },

            async addUser() {
                await fetch('/api/approved-users', {
                    method: 'POST',
                    headers: { ...this.authHeader, 'Content-Type': 'application/json' },
                    body: JSON.stringify({ github_username: this.newUsername })
                });
                this.newUsername = '';
                await this.loadUsers();
            },

            async removeUser(user) {
                if (!confirm(`Remove ${user.github_username}?`)) return;
                await fetch(`/api/approved-users/${user.github_username}`, {
                    method: 'DELETE',
                    headers: this.authHeader
                });
                await this.loadUsers();
            },

            getStatusClass(status) {
                if (!status) return 'text-gray-400';
                if (status.conclusion === 'success') return 'text-green-400';
                if (status.conclusion === 'failure') return 'text-red-400';
                if (status.status === 'in_progress') return 'text-yellow-400';
                return 'text-gray-400';
            },

            getStatusDotClass(status) {
                if (!status) return 'bg-gray-400';
                if (status.conclusion === 'success') return 'bg-green-400';
                if (status.conclusion === 'failure') return 'bg-red-400';
                if (status.status === 'in_progress') return 'bg-yellow-400 animate-pulse';
                return 'bg-gray-400';
            },

            getStatusText(status) {
                if (!status) return 'Unknown';
                if (status.conclusion) return status.conclusion;
                if (status.status) return status.status;
                return 'Unknown';
            }
        }
    }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
