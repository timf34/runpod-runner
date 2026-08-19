"""The one place ssh/scp argv are built. Every remote call in rp goes through here.

Pods are reached as ``root@<publicIp> -p <port>`` with the laptop's private key, host-key
checking disabled and a per-pod known_hosts file (pods get a fresh host key on every
(re)start, so the file is truncated on ``rp up`` / ``rp start``).
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import RpError


class SSHError(RpError):
    def __init__(self, msg: str, returncode: int | None = None):
        super().__init__(msg)
        self.returncode = returncode


@dataclass(frozen=True)
class Conn:
    host: str
    port: int
    identity: Path
    known_hosts: Path
    user: str = "root"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def _base_opts(conn: Conn, *, batch: bool = True, connect_timeout: int = 20) -> list[str]:
    opts = [
        "-i", str(conn.identity),
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={conn.known_hosts}",
        "-o", "LogLevel=ERROR",
        "-o", f"ConnectTimeout={connect_timeout}",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
    ]
    if batch:
        opts += ["-o", "BatchMode=yes"]
    return opts


def ssh_argv(
    conn: Conn,
    remote_cmd: str | None = None,
    *,
    tty: bool = False,
    batch: bool = True,
    connect_timeout: int = 20,
) -> list[str]:
    argv = ["ssh", *_base_opts(conn, batch=batch, connect_timeout=connect_timeout), "-p", str(conn.port)]
    if tty:
        argv.append("-t")
    argv += ["--", conn.target]
    if remote_cmd is not None:
        argv.append(remote_cmd)
    return argv


def scp_argv(conn: Conn, src: str, dst: str, *, recursive: bool = False) -> list[str]:
    argv = ["scp", *_base_opts(conn), "-P", str(conn.port)]
    if recursive:
        argv.append("-r")
    argv += ["--", src, dst]
    return argv


def remote_path(conn: Conn, path: str) -> str:
    return f"{conn.target}:{path}"


def run_ssh(
    conn: Conn,
    remote_cmd: str,
    *,
    check: bool = True,
    capture: bool = False,
    input: str | None = None,
    tty: bool = False,
    timeout: float | None = None,
    connect_timeout: int = 20,
) -> subprocess.CompletedProcess:
    """Run ``remote_cmd`` on the pod. Output streams to the terminal unless ``capture``."""
    argv = ssh_argv(conn, remote_cmd, tty=tty, connect_timeout=connect_timeout)
    try:
        # input=None -> stdin is inherited (needed for interactive-ish streaming); else piped.
        proc = subprocess.run(argv, text=True, input=input, capture_output=capture, timeout=timeout)
    except FileNotFoundError as e:
        raise RpError("ssh binary not found on this machine") from e
    except subprocess.TimeoutExpired as e:
        raise SSHError(f"ssh to {conn.host}:{conn.port} timed out after {timeout}s") from e
    if proc.returncode == 255:
        detail = (proc.stderr or "").strip() if capture else "see output above"
        raise SSHError(f"ssh to {conn.host}:{conn.port} failed (exit 255): {detail}", 255)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else "see output above"
        raise SSHError(
            f"remote command on {conn.host}:{conn.port} exited {proc.returncode}: {detail[:800]}",
            proc.returncode,
        )
    return proc


def run_scp(conn: Conn, src: str, dst: str, *, recursive: bool = False) -> None:
    argv = scp_argv(conn, src, dst, recursive=recursive)
    try:
        proc = subprocess.run(argv, text=True, capture_output=True)
    except FileNotFoundError as e:
        raise RpError("scp binary not found on this machine") from e
    if proc.returncode != 0:
        raise SSHError(f"scp {src} -> {dst} failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:800]}", proc.returncode)


def interactive_ssh(conn: Conn, remote_cmd: str | None = None) -> int:
    """Hand the terminal to ssh (interactive shell or a command with a tty)."""
    argv = ssh_argv(conn, remote_cmd, tty=sys.stdin.isatty(), batch=False)
    try:
        return subprocess.call(argv)
    except FileNotFoundError as e:
        raise RpError("ssh binary not found on this machine") from e


def wait_for_ssh(conn: Conn, *, timeout: float = 600, interval: float = 5, log=None) -> float:
    """Poll ``ssh ... true`` until it succeeds. Returns seconds waited; raises SSHError on timeout."""
    start = time.monotonic()
    last_err = ""
    while True:
        argv = ssh_argv(conn, "true", connect_timeout=10)
        try:
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=60)
        except FileNotFoundError as e:
            raise RpError("ssh binary not found on this machine") from e
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None and proc.returncode == 0:
            return time.monotonic() - start
        if proc is not None:
            last_err = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            last_err = last_err[0]
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise SSHError(
                f"ssh to {conn.host}:{conn.port} not reachable after {int(elapsed)}s (last: {last_err or 'timeout'})"
            )
        if log:
            log(f"  waiting for sshd ({int(elapsed)}s){': ' + last_err if last_err else ''}")
        time.sleep(interval)
