"""`rp bootstrap`: clone the experiment repo on the pod, install .env, build a venv, and
(optionally) register a per-pod GitHub deploy key so the pod can push without your token."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from . import RpError
from .config import Config, profiles_dir
from .ssh import Conn, remote_path, run_scp, run_ssh
from .state import PodState, save_state, utcnow
from .term import info

REMOTE_RP_DIR = "/workspace/.rp"
DEPLOY_KEY_PATH = "/root/.ssh/deploy_ed25519"
MIN_CUDA_RE = re.compile(r"^#\s*rp-min-cuda:\s*([0-9]+\.[0-9]+)", re.M)


# -- pure helpers (unit-tested) ------------------------------------------------

def parse_github_repo(url: str) -> str | None:
    """'https://github.com/o/r.git' | 'git@github.com:o/r.git' | 'github.com/o/r' -> 'o/r' (None if not GitHub)."""
    m = re.match(r"^(?:https?://|git@|ssh://git@)?(?:www\.)?github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def repo_dir_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].split(":")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    if not name:
        raise RpError(f"cannot derive a directory name from repo url {url!r}")
    return name


def parse_min_cuda(text: str) -> str | None:
    m = MIN_CUDA_RE.search(text)
    return m.group(1) if m else None


def max_cuda(versions: list[str]) -> str | None:
    vs = [v for v in versions if v]
    if not vs:
        return None
    return max(vs, key=lambda v: tuple(int(x) for x in v.split(".")))


def env_prefix(env: dict[str, str]) -> str:
    """KEY=val pairs, shell-quoted, for prefixing a remote command."""
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if v is not None)


def git_identity(cfg: Config) -> tuple[str | None, str | None]:
    name, email = cfg.git_name, cfg.git_email
    for attr, key in (("name", "user.name"), ("email", "user.email")):
        if (name if attr == "name" else email):
            continue
        try:
            out = subprocess.run(["git", "config", "--get", key], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            out = ""
        if attr == "name":
            name = out or None
        else:
            email = out or None
    return name, email


# -- gh deploy keys ------------------------------------------------------------

def _gh(*args: str) -> str:
    if not shutil.which("gh"):
        raise RpError("the GitHub CLI `gh` is required for --deploy-key (brew install gh && gh auth login)")
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RpError(f"gh {' '.join(args[:3])} ... failed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[:600]}")
    return proc.stdout


def ensure_deploy_key(conn: Conn, st: PodState, owner_repo: str) -> str:
    """Create an ed25519 key ON the pod (if missing), register it as a write deploy key on
    GitHub (if not already registered with the same public key), record it in the state file.
    Returns the git ssh command to use on the pod."""
    script = (
        "set -e; mkdir -p /root/.ssh && chmod 700 /root/.ssh; "
        f'[ -f {DEPLOY_KEY_PATH} ] || ssh-keygen -q -t ed25519 -N "" -f {DEPLOY_KEY_PATH} -C rp-{shlex.quote(st.name)}; '
        "grep -q '^github.com ssh-ed25519' /root/.ssh/known_hosts 2>/dev/null "
        "|| ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null; "
        f"cat {DEPLOY_KEY_PATH}.pub"
    )
    pub = run_ssh(conn, script, capture=True).stdout.strip().splitlines()[-1].strip()
    if not pub.startswith("ssh-ed25519"):
        raise RpError(f"unexpected deploy key output from pod: {pub[:80]!r}")
    pub_core = " ".join(pub.split()[:2])
    existing = st.deploy_key or {}
    if existing.get("repo") == owner_repo and existing.get("pub") == pub_core and existing.get("id"):
        info(f"deploy key already registered on {owner_repo} (id {existing['id']})")
    else:
        if existing.get("id"):
            info(f"replacing stale deploy key {existing.get('id')} on {existing.get('repo')}")
            delete_deploy_key(st, quiet=True)
        title = f"rp {st.name} {utcnow()}"
        out = _gh("api", f"repos/{owner_repo}/keys", "-f", f"title={title}", "-f", f"key={pub}", "-F", "read_only=false")
        try:
            key_id = json.loads(out)["id"]
        except (ValueError, KeyError) as e:
            raise RpError(f"could not parse deploy key id from gh output: {out[:300]}") from e
        st.deploy_key = {"repo": owner_repo, "id": key_id, "title": title, "pub": pub_core, "created": utcnow()}
        save_state(st)
        info(f"registered write deploy key id {key_id} on {owner_repo} (will be deleted by `rp down {st.name}`)")
    return f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


def delete_deploy_key(st: PodState, *, quiet: bool = False) -> bool:
    """Delete the pod's deploy key on GitHub. Returns True if gone (deleted or already absent)."""
    dk = st.deploy_key
    if not dk or not dk.get("id"):
        return True
    try:
        _gh("api", "-X", "DELETE", f"repos/{dk['repo']}/keys/{dk['id']}")
    except RpError as e:
        if "404" in str(e) or "Not Found" in str(e):
            pass
        else:
            info(f"warning: could not delete deploy key {dk['id']} on {dk['repo']}: {e}")
            return False
    if not quiet:
        info(f"deleted deploy key {dk['id']} on {dk['repo']}")
    st.deploy_key = None
    save_state(st)
    return True


# -- the command -----------------------------------------------------------------

def run_bootstrap(cfg: Config, st: PodState, conn: Conn, args) -> None:
    repo_url = args.repo or st.repo_url
    if not repo_url:
        raise RpError("--repo <url> is required the first time you bootstrap a pod")
    owner_repo = parse_github_repo(repo_url)
    repo_name = repo_dir_name(repo_url)
    repo_dir = args.dest or st.repo_dir or f"/workspace/{repo_name}"
    vn = args.venv_name or ""
    venv = vn if vn.startswith("/") else f"/workspace/{vn or repo_name}_venv"
    if st.venv and not vn and not args.dest:
        venv = st.venv  # keep whatever a previous bootstrap chose

    pdir = profiles_dir()
    script = pdir / "bootstrap.sh"
    if not script.exists():
        raise RpError(f"bootstrap.sh not found in {pdir} (set RP_PROFILES_DIR or reinstall with pip install -e .)")

    uploads: list[tuple[Path, str]] = [(script, f"{REMOTE_RP_DIR}/bootstrap.sh")]
    reqs_must: list[str] = []
    reqs_opt: list[str] = []
    min_cudas: list[str] = []
    for prof in args.profile or []:
        p = pdir / (prof if prof.endswith(".txt") else f"{prof}.txt")
        if not p.exists():
            avail = ", ".join(sorted(x.stem for x in pdir.glob("*.txt")))
            raise RpError(f"profile {prof!r} not found in {pdir} (available: {avail})")
        uploads.append((p, f"{REMOTE_RP_DIR}/req/{p.name}"))
        reqs_must.append(f"{REMOTE_RP_DIR}/req/{p.name}")
        min_cudas.append(parse_min_cuda(p.read_text()) or "")
    for r in args.req or []:
        lp = Path(r).expanduser()
        if lp.is_file():
            uploads.append((lp, f"{REMOTE_RP_DIR}/req/{lp.name}"))
            reqs_must.append(f"{REMOTE_RP_DIR}/req/{lp.name}")
            min_cudas.append(parse_min_cuda(lp.read_text()) or "")
        else:
            # repo-relative (or absolute) path on the pod
            reqs_must.append(r if r.startswith("/") else f"{repo_dir}/{r}")
    if not reqs_must and not args.no_req:
        reqs_opt.append(f"{repo_dir}/requirements.txt")

    dotenv_remote = ""
    if args.env:
        ep = Path(args.env).expanduser()
        if not ep.is_file():
            raise RpError(f".env file {ep} not found")
        uploads.append((ep, f"{REMOTE_RP_DIR}/dotenv"))
        dotenv_remote = f"{REMOTE_RP_DIR}/dotenv"

    git_ssh_cmd = ""
    clone_url = repo_url
    if args.deploy_key:
        if not owner_repo:
            raise RpError(f"--deploy-key needs a GitHub repo url, got {repo_url!r}")
        info(f"setting up a deploy key for {owner_repo} on pod {st.name} ...")
        git_ssh_cmd = ensure_deploy_key(conn, st, owner_repo)
        clone_url = f"git@github.com:{owner_repo}.git"
    git_name, git_email = git_identity(cfg)

    info(f"uploading {len(uploads)} file(s) to {conn.host}:{conn.port}:{REMOTE_RP_DIR} ...")
    run_ssh(conn, f"mkdir -p {REMOTE_RP_DIR}/req {REMOTE_RP_DIR}/jobs", capture=True)
    for local, remote in uploads:
        run_scp(conn, str(local), remote_path(conn, remote))
    if dotenv_remote:
        run_ssh(conn, f"chmod 600 {dotenv_remote}", capture=True)

    env = {
        "RP_REPO_URL": clone_url,
        "RP_REPO_DIR": repo_dir,
        "RP_BRANCH": args.branch or "",
        "RP_VENV": venv,
        "RP_REQS": " ".join(reqs_must),
        "RP_REQS_OPT": " ".join(reqs_opt),
        "RP_DOTENV": dotenv_remote,
        "RP_MIN_CUDA": max_cuda(min_cudas) or "",
        "RP_GIT_SSH_COMMAND": git_ssh_cmd,
        "RP_GIT_NAME": git_name or "",
        "RP_GIT_EMAIL": git_email or "",
        "RP_SKIP_SANITY": "1" if args.skip_sanity else "0",
    }
    remote_cmd = f"{env_prefix(env)} bash {REMOTE_RP_DIR}/bootstrap.sh"
    st.repo_url, st.repo_dir, st.venv = repo_url, repo_dir, venv
    save_state(st)
    info(f"running bootstrap on {st.name} (streaming; also logged to /workspace/bootstrap.log) ...")
    proc = run_ssh(conn, remote_cmd, check=False)
    if proc.returncode != 0:
        raise RpError(
            f"bootstrap on {st.name} failed with exit {proc.returncode} "
            f"(see above, or: rp ssh {st.name} -- tail -50 /workspace/bootstrap.log)"
        )
    info(
        f"\nbootstrap done. repo={repo_dir} venv={venv}\n"
        f"next: rp run {st.name} --job <name> -- python -u your_script.py ...   (runs in {repo_dir} with the venv active)"
    )
