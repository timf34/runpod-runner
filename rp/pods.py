"""Shared pod helpers: API client factory, state<->API refresh, polling, volume lookup."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from . import RpError
from .api import RunPodClient
from .config import Config, require_api_key
from .ssh import Conn
from .state import PodState, known_hosts_path, save_state

DC_RE = re.compile(r"^[A-Z]{2,4}-[A-Z]{2,4}-\d+$")  # e.g. EUR-IS-3, US-TX-3


def get_client(cfg: Config) -> RunPodClient:
    """Factory (monkeypatched in tests)."""
    return RunPodClient(require_api_key(cfg))


# -- timestamps ------------------------------------------------------------

def parse_runpod_ts(s: str | None) -> datetime | None:
    """RunPod returns e.g. '2026-08-19 13:15:43.832 +0000 UTC' (and sometimes ISO 8601)."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z UTC", "%Y-%m-%d %H:%M:%S %z UTC", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def human_age(ts: datetime | None, now: datetime | None = None) -> str:
    if ts is None:
        return "?"
    now = now or datetime.now(timezone.utc)
    secs = int((now - ts).total_seconds())
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
    return f"{secs // 86400}d{(secs % 86400) // 3600}h"


# -- pod dict helpers --------------------------------------------------------

def pod_status(pod: dict) -> str:
    return {"RUNNING": "running", "EXITED": "stopped", "TERMINATED": "terminated"}.get(
        str(pod.get("desiredStatus") or "").upper(), str(pod.get("desiredStatus") or "unknown").lower()
    )


def pod_ssh_endpoint(pod: dict) -> tuple[str | None, int | None]:
    ip = pod.get("publicIp") or None
    pm = pod.get("portMappings") or {}
    port = pm.get("22") if isinstance(pm, dict) else None
    try:
        port = int(port) if port is not None else None
    except (TypeError, ValueError):
        port = None
    return ip, port


def pod_gpu_label(pod: dict) -> str:
    machine = pod.get("machine") or {}
    gpu = machine.get("gpuTypeId") or (pod.get("gpu") or {}).get("displayName") or "?"
    n = pod.get("gpuCount") or (pod.get("gpu") or {}).get("count") or 1
    return f"{gpu} x{n}" if n and int(n) > 1 else str(gpu)


def pod_dc(pod: dict) -> str:
    return (pod.get("machine") or {}).get("dataCenterId") or (pod.get("networkVolume") or {}).get("dataCenterId") or "?"


def redact_pod(pod: dict) -> dict:
    """Drop the env block (it carries JUPYTER_PASSWORD) for --json output."""
    out = dict(pod)
    out.pop("env", None)
    return out


# -- state refresh -----------------------------------------------------------

def refresh_state(client: RunPodClient, st: PodState, *, save: bool = True) -> dict:
    """GET the pod and sync ip/port/status into the local state. Returns the pod dict."""
    pod = client.get_pod(st.id)
    ip, port = pod_ssh_endpoint(pod)
    st.ip, st.port = ip, port
    status = pod_status(pod)
    if status == "running" and (not ip or not port):
        status = "starting"
    st.status = status
    if save:
        save_state(st)
    return pod


def wait_for_network(client: RunPodClient, st: PodState, *, timeout: float = 600, interval: float = 5, log=None) -> dict:
    """Poll GET /pods/{id} until publicIp and the port-22 mapping are present."""
    start = time.monotonic()
    while True:
        pod = client.get_pod(st.id)
        ip, port = pod_ssh_endpoint(pod)
        if ip and port:
            st.ip, st.port = ip, port
            save_state(st)
            return pod
        status = str(pod.get("desiredStatus") or "?")
        if status.upper() in ("EXITED", "TERMINATED"):
            st.status = pod_status(pod)
            save_state(st)
            raise RpError(f"pod {st.id} went to {status} while waiting for its network (see `rp status {st.name}`)")
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise RpError(
                f"pod {st.id} has no public ip/port after {int(elapsed)}s (desiredStatus={status}). "
                f"It may still be scheduling; check later with `rp status {st.name}`."
            )
        if log:
            log(f"  waiting for ip/port ({int(elapsed)}s, status={status}, lastStatusChange={str(pod.get('lastStatusChange') or '')[:60]!r})")
        time.sleep(interval)


def connection(cfg: Config, st: PodState, *, client: RunPodClient | None = None) -> Conn:
    """Build the ssh Conn for a pod, refreshing ip/port from the API if missing."""
    if not st.ip or not st.port:
        client = client or get_client(cfg)
        refresh_state(client, st)
        if not st.ip or not st.port:
            raise RpError(f"pod {st.name} ({st.id}) has no public ip/port (status={st.status}); start it or wait")
    if not cfg.ssh_key.exists():
        raise RpError(f"ssh private key {cfg.ssh_key} not found (set defaults.ssh_key in config)")
    return Conn(host=st.ip, port=int(st.port), identity=cfg.ssh_key, known_hosts=known_hosts_path(st.name))


def reset_known_hosts(name: str) -> None:
    p = known_hosts_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


# -- volumes -----------------------------------------------------------------

def looks_like_dc(s: str) -> bool:
    return bool(DC_RE.match(s or ""))


def resolve_volume(client: RunPodClient, spec: str) -> dict:
    """``spec`` is a network volume id or a datacenter id. Returns the volume dict."""
    vols = client.list_volumes()
    if looks_like_dc(spec):
        hits = [v for v in vols if v.get("dataCenterId") == spec]
        if not hits:
            dcs = sorted({v.get("dataCenterId") for v in vols})
            raise RpError(f"no network volume in datacenter {spec}; you have volumes in: {', '.join(dcs)}")
        if len(hits) > 1:
            lines = "\n".join(f"  {v['id']}  {v.get('name')}  {v.get('size')}GB" for v in hits)
            raise RpError(f"{len(hits)} network volumes in {spec}; pass one by id:\n{lines}")
        return hits[0]
    for v in vols:
        if v.get("id") == spec:
            return v
    raise RpError(f"network volume {spec!r} not found (see `rp volumes`)")


def key_comment(pub_line: str) -> str:
    """'ssh-ed25519 AAAA... user@host' -> 'user@host' (or the key type + a few chars)."""
    parts = pub_line.split()
    if len(parts) >= 3:
        return " ".join(parts[2:])
    if len(parts) == 2:
        return f"{parts[0]} {parts[1][:12]}..."
    return pub_line[:20]
