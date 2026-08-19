"""Local pod state: ``~/.runpod-runner/pods/<name>.json`` (+ a per-pod known_hosts file)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import RpError
from .config import home_dir

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOB_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")  # tmux forbids '.' and ':' in session names


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise RpError(f"bad pod name {name!r}: use letters, digits, '.', '_' or '-' (max 64 chars)")
    return name


def validate_job(job: str) -> str:
    if not JOB_RE.match(job or ""):
        raise RpError(f"bad job name {job!r}: use letters, digits, '_' or '-' only (tmux forbids '.' and ':')")
    return job


@dataclass
class PodState:
    name: str
    id: str
    status: str = "creating"  # creating | running | stopped | terminated | unknown
    ip: str | None = None
    port: int | None = None
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    volume: str | None = None
    volume_dc: str | None = None
    gpu: list[str] = field(default_factory=list)
    gpu_count: int = 1
    image: str = ""
    keys: list[str] = field(default_factory=list)  # public-key comments/fingerprints injected at creation
    jupyter_password: str | None = None
    repo_url: str | None = None
    repo_dir: str | None = None
    venv: str | None = None
    deploy_key: dict | None = None  # {"repo": "owner/repo", "id": 123, "title": ..., "pub": ..., "created": ...}
    last_job: str | None = None
    jobs: dict = field(default_factory=dict)  # job -> {"cmd", "cwd", "started", "log"}
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PodState":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def ssh_target(self) -> str:
        return f"{self.ip}:{self.port}" if self.ip and self.port else "(no ip/port yet)"


def pods_dir() -> Path:
    return home_dir() / "pods"


def state_path(name: str) -> Path:
    return pods_dir() / f"{name}.json"


def known_hosts_path(name: str) -> Path:
    return pods_dir() / f"{name}.known_hosts"


def save_state(st: PodState) -> Path:
    st.updated = utcnow()
    path = state_path(st.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st.to_dict(), indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def state_exists(name: str) -> bool:
    return state_path(name).exists()


def load_state(name: str) -> PodState:
    validate_name(name)
    path = state_path(name)
    if not path.exists():
        raise RpError(
            f"no local pod named {name!r} in {pods_dir()} "
            f"(see `rp ls`; import an existing pod with `rp adopt <pod-id> --name {name}`)"
        )
    try:
        return PodState.from_dict(json.loads(path.read_text()))
    except (ValueError, TypeError) as e:
        raise RpError(f"corrupt state file {path}: {e}") from e


def list_states() -> list[PodState]:
    out: list[PodState] = []
    if not pods_dir().exists():
        return out
    for p in sorted(pods_dir().glob("*.json")):
        try:
            out.append(PodState.from_dict(json.loads(p.read_text())))
        except (ValueError, TypeError):
            continue
    return out


def delete_state(name: str) -> None:
    for p in (state_path(name), known_hosts_path(name)):
        if p.exists():
            p.unlink()
