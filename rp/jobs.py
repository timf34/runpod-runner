"""`rp run` / `rp logs` / `rp jobs` / `rp kill`: long-running commands in detached tmux sessions.

Each job J runs ``bash /workspace/.rp/jobs/J.sh > /workspace/J.log 2>&1; echo EXIT=$? >> /workspace/J.log``
inside tmux session J, so the log survives ssh disconnects and ends with an EXIT=<code> trailer.
"""

from __future__ import annotations

import shlex
from datetime import datetime, timezone

from . import RpError
from .ssh import Conn, run_ssh
from .state import PodState, save_state, utcnow, validate_job

JOBS_DIR = "/workspace/.rp/jobs"


def log_path(job: str) -> str:
    return f"/workspace/{job}.log"


def script_path(job: str) -> str:
    return f"{JOBS_DIR}/{job}.sh"


def default_job_name() -> str:
    return datetime.now(timezone.utc).strftime("job-%Y%m%d-%H%M%S")


def build_job_script(
    job: str,
    command: str,
    *,
    cwd: str,
    venv: str | None,
    dotenv: bool = False,
    extra_env: dict[str, str] | None = None,
) -> str:
    """The bash script that runs inside tmux. Exit status == the command's exit status."""
    lines = [
        "#!/usr/bin/env bash",
        f"# rp job {job} -- generated {utcnow()}",
        "set -o pipefail",
        f"cd {shlex.quote(cwd)} || {{ echo \"[rp] cannot cd to {cwd}\" >&2; exit 97; }}",
        "export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1",
    ]
    if venv:
        lines.append(
            f"if [ -f {shlex.quote(venv)}/bin/activate ]; then source {shlex.quote(venv)}/bin/activate; "
            f"else echo \"[rp] warning: venv {venv} not found; using system python\" >&2; fi"
        )
    if dotenv:
        lines.append("if [ -f .env ]; then set -a; source ./.env; set +a; fi")
    for k, v in (extra_env or {}).items():
        lines.append(f"export {k}={shlex.quote(v)}")
    lines += [
        f'echo "[rp] job {job} start $(date -u +%FT%TZ) host=$(hostname) cwd=$PWD python=$(command -v python)"',
        f"echo {shlex.quote('[rp] cmd: ' + command)}",
        command,
    ]
    return "\n".join(lines) + "\n"


def launch_remote_cmd(job: str) -> str:
    """Remote shell snippet: write the script from stdin, refuse duplicate sessions, start tmux."""
    log = log_path(job)
    script = script_path(job)
    return (
        f"set -e; mkdir -p {JOBS_DIR}; "
        "command -v tmux >/dev/null 2>&1 || { echo 'tmux is not installed on the pod (run rp bootstrap, or: apt-get install -y tmux)' >&2; exit 96; }; "
        f"if tmux has-session -t ={job} 2>/dev/null; then echo 'job {job} already has a live tmux session; choose another --job or rp kill it' >&2; exit 95; fi; "
        f"cat > {script}; chmod +x {script}; "
        f"if [ -f {log} ]; then mv {log} {log}.prev-$(date -u +%Y%m%d-%H%M%S); fi; "
        f"tmux new-session -d -s {job} 'bash {script} > {log} 2>&1; echo EXIT=$? >> {log}'; "
        f"echo started"
    )


def start_job(conn: Conn, st: PodState, *, job: str, command: str, cwd: str, venv: str | None, dotenv: bool, extra_env: dict) -> None:
    validate_job(job)
    if not command.strip():
        raise RpError("no command given (put it after `--`)")
    script = build_job_script(job, command, cwd=cwd, venv=venv, dotenv=dotenv, extra_env=extra_env)
    proc = run_ssh(conn, launch_remote_cmd(job), input=script, capture=True, check=False)
    if proc.returncode != 0:
        raise RpError(f"could not start job {job} on {st.name} (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[:600]}")
    st.last_job = job
    st.jobs[job] = {"cmd": command, "cwd": cwd, "started": utcnow(), "log": log_path(job)}
    save_state(st)


def tail_cmd(job: str, n: int, follow: bool) -> str:
    log = log_path(job)
    flags = f"-n {int(n)}" + (" -F" if follow else "")
    return f"if [ ! -f {log} ]; then echo 'no log yet at {log}' >&2; exit 2; fi; tail {flags} {log}"


def jobs_status_cmd(jobs: list[str]) -> str:
    """One ssh round-trip: tmux sessions + per-job liveness/exit status."""
    parts = ["echo '## tmux sessions'; tmux ls 2>/dev/null || echo '(none)'", "echo '## jobs'"]
    for j in jobs:
        log = log_path(j)
        parts.append(
            f"if tmux has-session -t ={j} 2>/dev/null; then s=running; else s=finished; fi; "
            f"if [ -f {log} ]; then last=$(tail -n 1 {log} 2>/dev/null | cut -c1-60); else last='(no log)'; fi; "
            + 'echo "' + j + ' $s ${last}"'
        )
    return "; ".join(parts)


def kill_cmd(job: str) -> str:
    log = log_path(job)
    return (
        f"if ! tmux has-session -t ={job} 2>/dev/null; then echo 'no live tmux session for job {job}' >&2; exit 3; fi; "
        f"pid=$(tmux list-panes -t ={job} -F '#{{pane_pid}}' | head -1); "
        "[ -n \"$pid\" ] && { kill -TERM -- -\"$pid\" 2>/dev/null || pkill -TERM -P \"$pid\" 2>/dev/null; }; sleep 1; "
        f"tmux kill-session -t ={job}; echo \"EXIT=KILLED $(date -u +%FT%TZ)\" >> {log}; echo killed {job}"
    )
