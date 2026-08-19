"""rp command line: up / ls / status / ssh / scp / volumes / bootstrap / run / logs / jobs / kill / down / start."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime, timezone

from . import RpError, __version__
from .bootstrap import delete_deploy_key, run_bootstrap
from .config import Config, load_config
from .gpus import alias_help, resolve_gpu
from .jobs import default_job_name, jobs_status_cmd, kill_cmd, log_path, start_job, tail_cmd
from . import pods
from .pods import (
    connection,
    human_age,
    key_comment,
    looks_like_dc,
    parse_runpod_ts,
    pod_dc,
    pod_gpu_label,
    pod_ssh_endpoint,
    pod_status,
    redact_pod,
    refresh_state,
    reset_known_hosts,
    resolve_volume,
    wait_for_network,
)
from .ssh import interactive_ssh, remote_path, run_scp, run_ssh, wait_for_ssh
from .state import (
    PodState,
    delete_state,
    list_states,
    load_state,
    save_state,
    state_exists,
    state_path,
    utcnow,
    validate_job,
    validate_name,
)
from .term import dim, green, info, red, table, yellow

DEFAULT_PORTS = ["22/tcp", "8888/http"]


# --------------------------------------------------------------------------- up

def build_create_body(cfg: Config, args, *, public_keys: list[str], volume: dict | None, jupyter_password: str) -> dict:
    gpu_ids = resolve_gpu(args.gpu or cfg.gpu, cfg.gpu_aliases)
    body = {
        "name": args.name,
        "cloudType": (args.cloud or cfg.cloud).upper(),
        "computeType": "GPU",
        "gpuCount": int(args.gpus),
        "gpuTypeIds": gpu_ids,
        "imageName": args.image or cfg.image,
        "containerDiskInGb": int(args.disk if args.disk is not None else cfg.disk),
        "volumeMountPath": "/workspace",
        "ports": list(DEFAULT_PORTS),
        "env": {"PUBLIC_KEY": "\n".join(public_keys), "JUPYTER_PASSWORD": jupyter_password},
    }
    if volume is not None:
        body["networkVolumeId"] = volume["id"]
        if volume.get("dataCenterId"):
            body["dataCenterIds"] = [volume["dataCenterId"]]
    else:
        body["volumeInGb"] = int(args.volume_size)
    if args.dc:
        dcs = [d.strip() for d in args.dc.split(",") if d.strip()]
        if volume is not None and volume.get("dataCenterId") and volume["dataCenterId"] not in dcs:
            raise RpError(f"--dc {args.dc} does not include the volume's datacenter {volume['dataCenterId']}")
        body["dataCenterIds"] = dcs
    for kv in args.env or []:
        if "=" not in kv:
            raise RpError(f"--env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        body["env"][k] = v
    return body


def cmd_up(cfg: Config, args) -> int:
    name = args.name or datetime.now(timezone.utc).strftime("rp-%Y%m%d-%H%M")
    args.name = validate_name(name)
    if state_exists(name):
        old = load_state(name)
        if old.status not in ("terminated",):
            raise RpError(
                f"a local pod named {name!r} already exists (id {old.id}, status {old.status}). "
                f"Pick another --name, or `rp down {name} --terminate`, or `rp forget {name}`."
            )
    public_keys = cfg.public_keys()
    jupyter_password = secrets.token_urlsafe(12)
    volume_spec = args.volume if args.volume is not None else cfg.volume
    if volume_spec and volume_spec.lower() in ("none", "no", "off", ""):
        volume_spec = None

    if args.dry_run:
        volume = None
        if volume_spec:
            if looks_like_dc(volume_spec):
                volume = {"id": f"<volume in {volume_spec}: resolved at run time>", "dataCenterId": volume_spec}
            else:
                volume = {"id": volume_spec, "dataCenterId": None}
                info(f"note: --dry-run does not call the API; the real run adds dataCenterIds=[<dc of {volume_spec}>]")
        body = build_create_body(cfg, args, public_keys=public_keys, volume=volume, jupyter_password=jupyter_password)
        print(json.dumps(body, indent=2))
        return 0

    client = pods.get_client(cfg)
    volume = resolve_volume(client, volume_spec) if volume_spec else None
    body = build_create_body(cfg, args, public_keys=public_keys, volume=volume, jupyter_password=jupyter_password)
    if volume is None:
        info(f"note: no network volume -> /workspace is a {body['volumeInGb']}GB pod volume that dies with the pod")
    info(
        f"creating pod {name}: gpu={body['gpuTypeIds'][0]}{'...' if len(body['gpuTypeIds']) > 1 else ''} x{body['gpuCount']} "
        f"image={body['imageName']} disk={body['containerDiskInGb']}GB "
        f"volume={volume['id'] + ' (' + volume.get('dataCenterId', '?') + ')' if volume else 'none'} "
        f"keys={[key_comment(k) for k in public_keys]}"
    )
    pod = client.create_pod(body)
    pod_id = pod.get("id")
    if not pod_id:
        raise RpError(f"POST /pods returned no id: {json.dumps(pod)[:500]}")
    st = PodState(
        name=name,
        id=pod_id,
        status="creating",
        volume=volume["id"] if volume else None,
        volume_dc=volume.get("dataCenterId") if volume else None,
        gpu=body["gpuTypeIds"],
        gpu_count=body["gpuCount"],
        image=body["imageName"],
        keys=[key_comment(k) for k in public_keys],
        jupyter_password=jupyter_password,
    )
    save_state(st)
    reset_known_hosts(name)
    cost = pod.get("costPerHr")
    info(f"created pod id={pod_id} {'$%.2f/h' % cost if cost else ''}  (state: {state_path(name)})")
    if args.no_wait:
        info(f"not waiting (--no-wait). Later: rp status {name}  /  rp ssh {name}")
        print(pod_id)
        return 0

    info("waiting for public ip/port ...")
    pod = wait_for_network(client, st, timeout=args.timeout, log=info)
    info(f"ip:port = {st.ip}:{st.port}; waiting for sshd ...")
    conn = connection(cfg, st, client=client)
    waited = wait_for_ssh(conn, timeout=args.timeout, log=info)
    st.status = "running"
    save_state(st)
    info(f"ssh is up after {int(waited)}s.")
    print(
        f"{green('pod ready')}: {name}  id={pod_id}  root@{st.ip} -p {st.port}  "
        f"gpu={pod_gpu_label(pod)}  {'$%.2f/h' % (pod.get('costPerHr') or cost or 0)}\n"
        f"  jupyter: https://{pod_id}-8888.proxy.runpod.net  (password: {jupyter_password})\n"
        f"  next:    rp bootstrap {name} --repo <git url> --env .env [--profile hf-latest]\n"
        f"  then:    rp run {name} --job J -- python -u script.py    |  rp logs {name} -f  |  rp down {name}"
    )
    return 0


# --------------------------------------------------------------------------- ls / status / volumes

def cmd_ls(cfg: Config, args) -> int:
    client = pods.get_client(cfg)
    pod_list = client.list_pods()
    local = {st.id: st.name for st in list_states()}
    if args.json:
        print(json.dumps([dict(redact_pod(p), localName=local.get(p.get("id"))) for p in pod_list], indent=2))
        return 0
    order = {"running": 0, "starting": 0, "stopped": 1, "terminated": 2}
    pod_list.sort(key=lambda p: (order.get(pod_status(p), 3), -(parse_runpod_ts(p.get("createdAt")) or datetime.min.replace(tzinfo=timezone.utc)).timestamp()))
    rows, styles = [], []
    now = datetime.now(timezone.utc)
    for p in pod_list:
        status = pod_status(p)
        ip, port = pod_ssh_endpoint(p)
        cost = p.get("costPerHr")
        rows.append([
            p.get("name") or "?",
            p.get("id") or "?",
            status.upper() if status == "running" else status,
            pod_gpu_label(p),
            pod_dc(p),
            f"{cost:.2f}" if isinstance(cost, (int, float)) else "?",
            f"{ip}:{port}" if ip and port else "-",
            human_age(parse_runpod_ts(p.get("createdAt")), now),
            local.get(p.get("id")) or "-",
        ])
        styles.append(green if status == "running" else (dim if status in ("stopped", "terminated") else yellow))
    if not rows:
        print("(no pods)")
        return 0
    print(table(["NAME", "ID", "STATUS", "GPU", "DC", "$/H", "SSH", "AGE", "LOCAL"], rows, styles))
    running = [p for p in pod_list if pod_status(p) == "running"]
    if running:
        burn = sum(float(p.get("costPerHr") or 0) for p in running)
        print(dim(f"{len(running)} running, ${burn:.2f}/h total"))
    return 0


def _resolve_state_or_id(cfg: Config, name_or_id: str, client=None) -> tuple[PodState | None, dict]:
    """For status: accept a local name, else a raw pod id."""
    if state_exists(name_or_id):
        st = load_state(name_or_id)
        client = client or pods.get_client(cfg)
        pod = refresh_state(client, st)
        return st, pod
    client = client or pods.get_client(cfg)
    return None, client.get_pod(name_or_id)


def cmd_status(cfg: Config, args) -> int:
    st, pod = _resolve_state_or_id(cfg, args.name)
    if args.json:
        print(json.dumps({"state": st.to_dict() if st else None, "pod": redact_pod(pod)}, indent=2))
        return 0
    status = pod_status(pod)
    ip, port = pod_ssh_endpoint(pod)
    colour = green if status == "running" else dim
    lines = [
        f"{'name':12} {pod.get('name')}" + (f"  (local: {st.name})" if st else "  (no local state; `rp adopt {id}`)".format(id=pod.get("id"))),
        f"{'id':12} {pod.get('id')}",
        f"{'status':12} {colour(status)}   {dim(str(pod.get('lastStatusChange') or ''))}",
        f"{'gpu':12} {pod_gpu_label(pod)}   dc={pod_dc(pod)}   ${pod.get('costPerHr')}/h",
        f"{'image':12} {pod.get('imageName')}",
        f"{'ssh':12} " + (f"ssh root@{ip} -p {port}" if ip and port else "-"),
        f"{'volume':12} " + (f"{(pod.get('networkVolume') or {}).get('id')} {(pod.get('networkVolume') or {}).get('name', '')} {(pod.get('networkVolume') or {}).get('size', '')}GB at {pod.get('volumeMountPath')}" if pod.get("networkVolumeId") else f"pod volume {pod.get('volumeInGb')}GB at {pod.get('volumeMountPath')}"),
        f"{'disk':12} container {pod.get('containerDiskInGb')}GB",
        f"{'created':12} {pod.get('createdAt')}  (age {human_age(parse_runpod_ts(pod.get('createdAt')))})",
    ]
    if st:
        lines.append(f"{'repo':12} {st.repo_dir or '-'}   venv {st.venv or '-'}")
        lines.append(f"{'deploy key':12} " + (f"{st.deploy_key['repo']} id {st.deploy_key['id']}" if st.deploy_key else "-"))
        lines.append(f"{'last job':12} {st.last_job or '-'}" + (f"   ({len(st.jobs)} job(s) recorded)" if st.jobs else ""))
        lines.append(f"{'jupyter':12} https://{pod.get('id')}-8888.proxy.runpod.net" + (f"  password {st.jupyter_password}" if st.jupyter_password else ""))
        lines.append(f"{'state file':12} {state_path(st.name)}")
    print("\n".join(lines))
    return 0


def cmd_volumes(cfg: Config, args) -> int:
    vols = pods.get_client(cfg).list_volumes()
    if args.json:
        print(json.dumps(vols, indent=2))
        return 0
    vols.sort(key=lambda v: (v.get("dataCenterId") or "", v.get("name") or ""))
    rows = [[v.get("id"), v.get("name"), v.get("dataCenterId"), f"{v.get('size')}GB"] for v in vols]
    print(table(["ID", "NAME", "DC", "SIZE"], rows) if rows else "(no network volumes)")
    return 0


def cmd_gpus(cfg: Config, args) -> int:
    print("GPU aliases (--gpu): first available type wins\n" + alias_help(cfg.gpu_aliases))
    print("\nAnything else is passed through as exact gpuTypeId(s), comma-separated.")
    return 0


# --------------------------------------------------------------------------- ssh / scp

def cmd_ssh(cfg: Config, args) -> int:
    st = load_state(args.name)
    conn = connection(cfg, st)
    cmd = " ".join(args.rest) if args.rest else None
    if cmd is None:
        return interactive_ssh(conn)
    proc = run_ssh(conn, cmd, check=False)
    return proc.returncode


def _scp_side(conn, spec: str) -> tuple[str, bool]:
    if spec.startswith("pod:"):
        return remote_path(conn, spec[len("pod:"):] or "."), True
    return spec, False


def cmd_scp(cfg: Config, args) -> int:
    st = load_state(args.name)
    conn = connection(cfg, st)
    src, src_remote = _scp_side(conn, args.src)
    dst, dst_remote = _scp_side(conn, args.dst)
    if src_remote == dst_remote:
        raise RpError("exactly one of <src>/<dst> must have the `pod:` prefix, e.g. rp scp mypod .env pod:/workspace/repo/.env")
    run_scp(conn, src, dst, recursive=args.recursive)
    info(f"copied {args.src} -> {args.dst}")
    return 0


# --------------------------------------------------------------------------- bootstrap / run / logs / jobs / kill

def cmd_bootstrap(cfg: Config, args) -> int:
    st = load_state(args.name)
    conn = connection(cfg, st)
    run_bootstrap(cfg, st, conn, args)
    return 0


def _parse_kv(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in items or []:
        if "=" not in kv:
            raise RpError(f"--env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def cmd_run(cfg: Config, args) -> int:
    st = load_state(args.name)
    conn = connection(cfg, st)
    if not args.rest:
        raise RpError("no command given; put it after `--`, e.g. rp run NAME --job J -- python -u script.py")
    command = " ".join(args.rest)
    job = validate_job(args.job or default_job_name())
    cwd = args.cwd or st.repo_dir or "/workspace"
    venv = None if args.no_venv else (args.venv or st.venv)
    start_job(conn, st, job=job, command=command, cwd=cwd, venv=venv, dotenv=args.dotenv, extra_env=_parse_kv(args.env))
    print(
        f"{green('started')} job {job} on {st.name} (tmux session {job}, cwd {cwd}{', venv ' + venv if venv else ''})\n"
        f"  follow:  rp logs {st.name} --job {job} -f\n"
        f"  status:  rp jobs {st.name}        kill: rp kill {st.name} --job {job}\n"
        f"  log:     {log_path(job)} (ends with EXIT=<code> when done)"
    )
    return 0


def _job_or_last(st: PodState, job: str | None) -> str:
    job = job or st.last_job
    if not job:
        raise RpError(f"no --job given and no job recorded for {st.name} yet (rp run {st.name} --job J -- ...)")
    return validate_job(job)


def cmd_logs(cfg: Config, args) -> int:
    st = load_state(args.name)
    job = _job_or_last(st, args.job)
    conn = connection(cfg, st)
    try:
        proc = run_ssh(conn, tail_cmd(job, args.lines, args.follow), check=False)
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def cmd_jobs(cfg: Config, args) -> int:
    st = load_state(args.name)
    conn = connection(cfg, st)
    jobs = list(st.jobs.keys())
    proc = run_ssh(conn, jobs_status_cmd(jobs), capture=True, check=False)
    out = (proc.stdout or "").rstrip()
    if proc.returncode != 0 and not out:
        raise RpError(f"could not query jobs on {st.name}: {(proc.stderr or '').strip()[:400]}")
    known = {j: st.jobs[j] for j in jobs}
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2 and parts[0] in known:
            meta = known[parts[0]]
            state = parts[1]
            tail = parts[2] if len(parts) > 2 else ""
            if state == "running":
                print(f"{green('running ')} {parts[0]:24} started {meta.get('started')}  {dim(tail)}")
            else:
                colour = green if tail.startswith("EXIT=0") else red if tail.startswith("EXIT=") else yellow
                print(f"{colour('finished')} {parts[0]:24} started {meta.get('started')}  {tail}")
            print(dim(f"          cmd: {meta.get('cmd')}  (cwd {meta.get('cwd')})"))
        else:
            print(line)
    return 0


def cmd_kill(cfg: Config, args) -> int:
    st = load_state(args.name)
    job = _job_or_last(st, args.job)
    conn = connection(cfg, st)
    proc = run_ssh(conn, kill_cmd(job), check=False)
    if proc.returncode == 3:
        info(f"job {job} has no live tmux session on {st.name} (already finished?) -- see rp jobs {st.name}")
        return 1
    return proc.returncode


# --------------------------------------------------------------------------- down / start / adopt / forget

def cmd_down(cfg: Config, args) -> int:
    st = load_state(args.name)
    client = pods.get_client(cfg)
    action = "terminate" if args.terminate else "stop"
    adopted = bool((st.notes or {}).get("adopted"))
    if not args.yes:
        # Guard rails: terminating, or touching a pod rp did not create, needs an explicit yes.
        reason = None
        if args.terminate:
            reason = f"DELETE pod {st.name} ({st.id}) permanently? Container disk is lost; network-volume data survives."
        elif adopted:
            reason = f"pod {st.name} ({st.id}) was adopted, not created by `rp up` -- someone else's job may be running on it. {action} it?"
        if reason:
            if not sys.stdin.isatty():
                raise RpError(f"{reason} Re-run with --yes to confirm (non-interactive).")
            ans = input(f"{reason} [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                info("aborted")
                return 1
    if st.deploy_key:
        delete_deploy_key(st)
    try:
        pod = client.get_pod(st.id)
    except RpError as e:
        pod = None
        info(f"warning: could not GET pod before {action}: {e}")
    current = pod_status(pod) if pod else "unknown"
    if args.terminate:
        client.delete_pod(st.id)
        st.status = "terminated"
        st.ip, st.port = None, None
        save_state(st)
        print(f"{red('terminated')} pod {st.name} ({st.id}). Local state kept at {state_path(st.name)} (rp forget {st.name} to drop it).")
        return 0
    if current == "stopped":
        info(f"pod {st.name} is already stopped")
    else:
        client.stop_pod(st.id)
    st.status = "stopped"
    st.ip, st.port = None, None
    save_state(st)
    vol_note = "(/workspace on the network volume persists; the container disk is wiped)" if st.volume else (
        yellow("(no network volume: the container disk is wiped; the pod volume persists until you terminate)"))
    print(f"{yellow('stopped')} pod {st.name} ({st.id}) {vol_note}. Resume: rp start {st.name}   Delete: rp down {st.name} --terminate")
    return 0


def cmd_start(cfg: Config, args) -> int:
    st = load_state(args.name)
    client = pods.get_client(cfg)
    pod = client.get_pod(st.id)
    if pod_status(pod) == "terminated":
        raise RpError(f"pod {st.name} ({st.id}) is terminated; create a new one with rp up")
    if pod_status(pod) != "running":
        info(f"starting pod {st.name} ({st.id}) ...")
        client.start_pod(st.id)
    else:
        info(f"pod {st.name} is already running; refreshing ip/port")
    reset_known_hosts(st.name)
    wait_for_network(client, st, timeout=args.timeout, log=info)
    conn = connection(cfg, st, client=client)
    wait_for_ssh(conn, timeout=args.timeout, log=info)
    st.status = "running"
    save_state(st)
    print(
        f"{green('running')}: {st.name} root@{st.ip} -p {st.port}\n"
        f"  note: the container disk was reset (apt packages, /root, tmux sessions gone); /workspace (repo, venv, HF cache) survived.\n"
        f"  `rp bootstrap {st.name}` is idempotent and safe to re-run (pulls + reinstalls tmux + checks the venv)."
    )
    return 0


def cmd_adopt(cfg: Config, args) -> int:
    client = pods.get_client(cfg)
    pod = client.get_pod(args.pod_id)
    name = validate_name(args.name or (pod.get("name") or pod["id"]).replace(" ", "-"))
    if state_exists(name) and not args.force:
        raise RpError(f"local pod {name!r} already exists (use --name or --force)")
    ip, port = pod_ssh_endpoint(pod)
    st = PodState(
        name=name,
        id=pod["id"],
        status=pod_status(pod),
        ip=ip,
        port=port,
        volume=pod.get("networkVolumeId") or None,
        volume_dc=(pod.get("networkVolume") or {}).get("dataCenterId"),
        gpu=[(pod.get("machine") or {}).get("gpuTypeId")] if (pod.get("machine") or {}).get("gpuTypeId") else [],
        gpu_count=int(pod.get("gpuCount") or 1),
        image=pod.get("imageName") or "",
        keys=["(adopted: keys unknown; ssh works only if your key was injected)"],
        notes={"adopted": utcnow()},
    )
    save_state(st)
    reset_known_hosts(name)
    print(f"adopted pod {pod['id']} as {name} ({st.status}, {st.ssh_target()}); state at {state_path(name)}")
    return 0


def cmd_forget(cfg: Config, args) -> int:
    st = load_state(args.name)
    if st.status in ("running", "starting", "creating") and not args.force:
        raise RpError(f"pod {st.name} ({st.id}) is {st.status}; rp down it first, or pass --force to drop only the local record")
    if st.deploy_key:
        info(f"note: deploy key {st.deploy_key.get('id')} on {st.deploy_key.get('repo')} is still registered")
    delete_state(st.name)
    print(f"forgot local state for {st.name} (pod {st.id} untouched)")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rp",
        description="Provision and drive RunPod GPU pods: rp up -> rp bootstrap -> rp run -> rp logs -> rp down",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"rp {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    def add(name, func, help, **kw):
        sp = sub.add_parser(name, help=help, description=help, formatter_class=argparse.RawDescriptionHelpFormatter, **kw)
        sp.set_defaults(func=func)
        return sp

    sp = add("up", cmd_up, "create a pod and wait until ssh answers")
    sp.add_argument("--name", help="local + RunPod pod name (default rp-<timestamp>)")
    sp.add_argument("--gpu", help="alias (h100|a100|h200|...; see `rp gpus`) or exact gpuTypeId(s), comma-separated")
    sp.add_argument("--gpus", type=int, default=1, help="GPU count (default 1)")
    sp.add_argument("--volume", help="network volume id, or a datacenter id like EUR-IS-3, or 'none'")
    sp.add_argument("--volume-size", type=int, default=50, help="pod volume GB when no network volume (default 50)")
    sp.add_argument("--disk", type=int, help="container disk GB (default 60)")
    sp.add_argument("--image", help="docker image (default runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)")
    sp.add_argument("--cloud", choices=["SECURE", "COMMUNITY", "secure", "community"], help="cloud type (default SECURE)")
    sp.add_argument("--dc", help="force dataCenterIds (comma-separated); normally derived from the volume")
    sp.add_argument("--env", action="append", metavar="KEY=VAL", help="extra container env var (repeatable)")
    sp.add_argument("--timeout", type=int, default=600, help="seconds to wait for ip/ssh (default 600)")
    sp.add_argument("--no-wait", action="store_true", help="return right after creation")
    sp.add_argument("--dry-run", action="store_true", help="print the POST /pods body and exit (no API calls)")

    sp = add("ls", cmd_ls, "list all pods on the account")
    sp.add_argument("--json", action="store_true")
    sp = add("status", cmd_status, "show one pod (local name or raw pod id) and refresh its ip/port")
    sp.add_argument("name")
    sp.add_argument("--json", action="store_true")
    sp = add("volumes", cmd_volumes, "list network volumes")
    sp.add_argument("--json", action="store_true")
    add("gpus", cmd_gpus, "show GPU aliases")

    sp = add("ssh", cmd_ssh, "interactive shell, or run a command: rp ssh NAME -- nvidia-smi")
    sp.add_argument("name")
    sp.epilog = "Everything after `--` is joined with spaces and run by the remote shell (so pipes/&& work)."
    sp = add("scp", cmd_scp, "copy files; prefix the pod side with pod: (rp scp NAME .env pod:/workspace/repo/.env)")
    sp.add_argument("name")
    sp.add_argument("src")
    sp.add_argument("dst")
    sp.add_argument("-r", "--recursive", action="store_true")

    sp = add("bootstrap", cmd_bootstrap, "clone repo, install .env, build venv + requirements on the pod (idempotent)")
    sp.add_argument("name")
    sp.add_argument("--repo", help="git clone url (required the first time)")
    sp.add_argument("--branch")
    sp.add_argument("--dest", help="clone dir on the pod (default /workspace/<repo>)")
    sp.add_argument("--env", help="local .env to install as <repo>/.env (mode 600)")
    sp.add_argument("--req", action="append", metavar="FILE", help="requirements file: a local file (uploaded) or a repo-relative path on the pod; repeatable. Default: <repo>/requirements.txt if present")
    sp.add_argument("--profile", action="append", metavar="NAME", help="built-in profile from profiles/ (hf-latest, vllm-0.11, vllm-cu124); repeatable")
    sp.add_argument("--no-req", action="store_true", help="do not install any requirements")
    sp.add_argument("--venv-name", help="venv at /workspace/<name>_venv (default <repo>_venv); absolute paths used as-is")
    sp.add_argument("--deploy-key", action="store_true", help="create a key on the pod + register a write deploy key on GitHub (gh) so the pod can push")
    sp.add_argument("--skip-sanity", action="store_true", help="skip the torch.cuda.is_available() check")

    sp = add("run", cmd_run, "start a command in a detached tmux session with a log + EXIT trailer")
    sp.add_argument("name")
    sp.add_argument("--job", help="job/tmux session name (default job-<timestamp>)")
    sp.add_argument("--cwd", help="working dir (default the bootstrapped repo, else /workspace)")
    sp.add_argument("--venv", help="venv to activate (default the bootstrapped one)")
    sp.add_argument("--no-venv", action="store_true")
    sp.add_argument("--dotenv", action="store_true", help="`set -a; source .env` in cwd before the command")
    sp.add_argument("--env", action="append", metavar="KEY=VAL", help="export extra env var (repeatable)")
    sp.epilog = "The command goes after `--`: rp run NAME --job J -- python -u train.py --epochs 3"
    sp = add("logs", cmd_logs, "tail a job's log")
    sp.add_argument("name")
    sp.add_argument("--job", help="default: last job started with rp run")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("-n", "--lines", type=int, default=50)
    sp = add("jobs", cmd_jobs, "list tmux sessions and the status of known jobs")
    sp.add_argument("name")
    sp = add("kill", cmd_kill, "kill a job's process group and tmux session")
    sp.add_argument("name")
    sp.add_argument("--job", help="default: last job")

    sp = add("down", cmd_down, "stop (default) or --terminate a pod; removes its deploy key; keeps local state")
    sp.add_argument("name")
    sp.add_argument("--terminate", action="store_true", help="DELETE the pod instead of stopping it")
    sp.add_argument("-y", "--yes", action="store_true", help="no confirmation prompt for --terminate")
    sp = add("start", cmd_start, "resume a stopped pod and re-resolve ip/port")
    sp.add_argument("name")
    sp.add_argument("--timeout", type=int, default=600)
    sp = add("adopt", cmd_adopt, "import an existing pod (e.g. one made in the web UI) into local state")
    sp.add_argument("pod_id")
    sp.add_argument("--name")
    sp.add_argument("--force", action="store_true")
    sp = add("forget", cmd_forget, "delete the local state file (does not touch the pod)")
    sp.add_argument("name")
    sp.add_argument("--force", action="store_true")
    return p


def split_rest(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split at the first standalone `--`: (args for argparse, raw remote command words)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, rest = split_rest(argv)
    parser = build_parser()
    args = parser.parse_args(head)
    args.rest = rest
    try:
        cfg = load_config()
        return int(args.func(cfg, args) or 0)
    except RpError as e:
        print(f"rp: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nrp: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
