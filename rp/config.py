"""Configuration: API key + defaults.

The API key is read from the environment variable ``RUNPOD_API_KEY`` or, failing that,
from ``~/.config/runpod-runner/config.toml`` (which should be mode 600)::

    api_key = "rpa_..."

    [defaults]
    gpu = "h100"                       # alias or exact gpuTypeId(s), comma-separated
    image = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    disk = 60                          # container disk GB
    volume = "cdv10pb3cq"              # network volume id, or a datacenter id like EUR-IS-3
    cloud = "SECURE"
    ssh_key = "~/.ssh/id_ed25519"      # private key used for ssh; its .pub is injected into pods
    extra_public_keys = ["ssh-rsa AAAA... runpodctl", "~/.ssh/other.pub"]
    git_name = "Your Name"             # used for commits made on the pod (deploy-key flow)
    git_email = "you@example.com"

    [gpus]                             # extra / overriding GPU aliases
    cheap = ["NVIDIA A40", "NVIDIA L40S"]

Environment overrides used mainly by tests: ``RP_CONFIG`` (config file path),
``RP_HOME`` (state dir, default ``~/.runpod-runner``), ``RP_PROFILES_DIR``.
"""

from __future__ import annotations

import os
import stat
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import RpError

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_CONFIG_PATH = "~/.config/runpod-runner/config.toml"
DEFAULT_HOME = "~/.runpod-runner"


def config_path() -> Path:
    return Path(os.environ.get("RP_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()


def home_dir() -> Path:
    return Path(os.environ.get("RP_HOME") or DEFAULT_HOME).expanduser()


def profiles_dir() -> Path:
    """Directory holding requirements profiles and bootstrap.sh (repo-level ``profiles/``)."""
    env = os.environ.get("RP_PROFILES_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "profiles"


@dataclass
class Config:
    api_key: str | None = None
    ssh_key: Path = field(default_factory=lambda: Path("~/.ssh/id_ed25519").expanduser())
    extra_public_keys: list[str] = field(default_factory=list)
    gpu: str = "h100"
    image: str = DEFAULT_IMAGE
    disk: int = 60
    volume: str | None = None
    cloud: str = "SECURE"
    git_name: str | None = None
    git_email: str | None = None
    gpu_aliases: dict[str, list[str]] = field(default_factory=dict)
    source: str = "defaults"

    def public_keys(self) -> list[str]:
        """All public key lines to inject into a new pod (laptop key first)."""
        keys: list[str] = []
        pub = self.ssh_key.with_suffix(self.ssh_key.suffix + ".pub")
        if not pub.exists():
            raise RpError(
                f"public key {pub} not found (set ssh_key in {config_path()} or create one with ssh-keygen -t ed25519)"
            )
        keys.append(pub.read_text().strip())
        for item in self.extra_public_keys:
            item = item.strip()
            if not item:
                continue
            if item.split()[0].startswith(("ssh-", "ecdsa-", "sk-")):
                keys.append(item)
            else:
                p = Path(item).expanduser()
                if not p.exists():
                    raise RpError(f"extra_public_keys entry {item!r} is neither a key line nor an existing file")
                keys.append(p.read_text().strip())
        # de-duplicate while keeping order
        seen: set[str] = set()
        out: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    data: dict = {}
    source = "env/defaults"
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            print(f"rp: warning: {path} is mode {mode:03o}; it holds your API key -- run: chmod 600 {path}", file=sys.stderr)
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise RpError(f"bad TOML in {path}: {e}") from e
        source = str(path)
    d = data.get("defaults", {}) or {}
    cfg = Config(
        api_key=os.environ.get("RUNPOD_API_KEY") or data.get("api_key") or None,
        ssh_key=Path(d.get("ssh_key", "~/.ssh/id_ed25519")).expanduser(),
        extra_public_keys=list(d.get("extra_public_keys", []) or []),
        gpu=str(d.get("gpu", "h100")),
        image=str(d.get("image", DEFAULT_IMAGE)),
        disk=int(d.get("disk", 60)),
        volume=(str(d["volume"]) if d.get("volume") else None),
        cloud=str(d.get("cloud", "SECURE")).upper(),
        git_name=d.get("git_name"),
        git_email=d.get("git_email"),
        gpu_aliases={str(k).lower(): list(v) for k, v in (data.get("gpus", {}) or {}).items()},
        source=source,
    )
    return cfg


def require_api_key(cfg: Config) -> str:
    if not cfg.api_key:
        raise RpError(
            "no RunPod API key. Export RUNPOD_API_KEY=... or put `api_key = \"...\"` in "
            f"{config_path()} (then chmod 600 it)."
        )
    return cfg.api_key
