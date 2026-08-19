"""Run profiles/bootstrap.sh locally (offline) against a fake nvidia-smi, a local git repo and
an empty requirements file, to exercise the driver check, clone/pull idempotency, .env install
and venv creation. Needs bash + git (no network; the pip self-upgrade is disabled)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "profiles" / "bootstrap.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash") or not shutil.which("git"),
    reason="needs bash + git",
)


def _fake_nvidia_smi(dir: Path, cuda: str) -> None:
    sh = dir / "nvidia-smi"
    sh.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "--query-gpu=name,memory.total,driver_version" ]; then echo "NVIDIA H100 80GB HBM3, 81559 MiB, 580.65"; exit 0; fi\n'
        f'echo "| NVIDIA-SMI 580.65    Driver Version: 580.65    CUDA Version: {cuda}     |"\n'
    )
    sh.chmod(0o755)


def _run(tmp: Path, ws: Path, env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{tmp / 'bin'}:{env['PATH']}"
    env.update({
        "RP_WORKSPACE": str(ws),
        "RP_BASHRC": str(tmp / "bashrc"),
        "RP_PIP_UPGRADE": "0",
        "RP_SKIP_SANITY": "0",
        "RP_REPO_DIR": str(ws / "Repo"),
        "RP_VENV": str(ws / "Repo_venv"),
        "RP_BRANCH": "",
        "RP_REQS": "",
        "RP_REQS_OPT": "",
        "RP_DOTENV": "",
        "RP_MIN_CUDA": "",
        "RP_GIT_SSH_COMMAND": "",
        "RP_GIT_NAME": "Test Bot",
        "RP_GIT_EMAIL": "bot@example.com",
    })
    env.update(env_extra)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "bin").mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    # a tiny local git repo to clone from
    src = tmp_path / "src-repo"
    src.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
    (src / "README.md").write_text("hi\n")
    (src / "requirements.txt").write_text("# intentionally empty\n")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], check=True)
    return tmp_path, ws, src


def test_driver_too_old_fails_loudly(sandbox):
    tmp, ws, src = sandbox
    _fake_nvidia_smi(tmp / "bin", "12.4")
    proc = _run(tmp, ws, {"RP_REPO_URL": str(src), "RP_MIN_CUDA": "12.8"})
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "driver max CUDA: 12.4" in proc.stdout
    assert "need CUDA >= 12.8" in proc.stdout and "only supports CUDA 12.4" in proc.stdout
    assert not (ws / "Repo").exists()  # failed before cloning
    assert "FAILED" in (ws / "bootstrap.log").read_text()


def test_full_run_then_idempotent_rerun(sandbox):
    tmp, ws, src = sandbox
    _fake_nvidia_smi(tmp / "bin", "12.8")
    dotenv = ws / ".rp" / "dotenv"
    dotenv.parent.mkdir(parents=True)
    dotenv.write_text("SECRET=1\n")
    proc = _run(tmp, ws, {"RP_REPO_URL": str(src), "RP_MIN_CUDA": "12.8", "RP_DOTENV": str(dotenv),
                          "RP_REQS_OPT": str(ws / "Repo" / "requirements.txt")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "driver CUDA 12.8 >= required 12.8: ok" in out
    assert (ws / "Repo" / "README.md").exists()
    envf = ws / "Repo" / ".env"
    assert envf.read_text() == "SECRET=1\n" and oct(envf.stat().st_mode & 0o777) == "0o600"
    assert not dotenv.exists()  # moved, not copied
    assert (ws / "Repo_venv" / "bin" / "python").exists()
    assert "pip install -r" in out and "torch not installed -- skipping CUDA check" in out
    assert (ws / "hf").is_dir()
    assert "HF_HOME=" in (tmp / "bashrc").read_text()
    assert "rp bootstrap OK" in out
    name = subprocess.run(["git", "-C", str(ws / "Repo"), "config", "user.name"], capture_output=True, text=True).stdout.strip()
    assert name == "Test Bot"

    # re-run: pull instead of clone, keep .env, reuse venv
    proc2 = _run(tmp, ws, {"RP_REPO_URL": str(src), "RP_MIN_CUDA": "12.8"})
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    assert "already cloned; git pull --ff-only" in proc2.stdout
    assert "keeping existing" in proc2.stdout
    assert "reusing existing venv" in proc2.stdout
    assert envf.read_text() == "SECRET=1\n"


def test_missing_required_reqs_file(sandbox):
    tmp, ws, src = sandbox
    _fake_nvidia_smi(tmp / "bin", "12.8")
    proc = _run(tmp, ws, {"RP_REPO_URL": str(src), "RP_REQS": str(ws / "nope.txt")})
    assert proc.returncode == 5 and "requirements file" in proc.stdout and "not found" in proc.stdout
