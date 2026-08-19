"""CLI flows with a fake RunPod client and a scripted fake ssh/scp (no network, no real pods)."""

from __future__ import annotations

import json

import rp.bootstrap
from conftest import read_state, run_cli


# -- up --------------------------------------------------------------------------

def test_up_dry_run_needs_no_key(env, monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY")
    rc, out, err = run_cli(["up", "--name", "d1", "--gpu", "h100", "--volume", "EUR-IS-3", "--disk", "80", "--dry-run"], capsys)
    assert rc == 0
    body = json.loads(out)
    assert body["name"] == "d1"
    assert body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe"]
    assert body["cloudType"] == "SECURE" and body["computeType"] == "GPU" and body["gpuCount"] == 1
    assert body["containerDiskInGb"] == 80 and body["volumeMountPath"] == "/workspace"
    assert body["ports"] == ["22/tcp", "8888/http"]
    assert body["dataCenterIds"] == ["EUR-IS-3"]
    assert "volumeInGb" not in body
    assert body["env"]["PUBLIC_KEY"].splitlines() == ["ssh-ed25519 AAAAC3FAKEKEY laptop@test", "ssh-rsa AAAAB3FAKERSA runpodctl"]
    assert len(body["env"]["JUPYTER_PASSWORD"]) >= 12
    assert not (env["home"] / "pods").exists()  # nothing saved


def test_up_dry_run_without_volume_uses_pod_volume(env, capsys):
    rc, out, _ = run_cli(["up", "--name", "d2", "--volume", "none", "--volume-size", "30", "--dry-run"], capsys)
    body = json.loads(out)
    assert body["volumeInGb"] == 30 and "networkVolumeId" not in body


def test_up_creates_polls_and_saves_state(env, client, fake_ssh, capsys):
    rc, out, err = run_cli(["up", "--name", "exp1", "--gpu", "h200", "--volume", "cdv10pb3cq", "--gpus", "2"], capsys)
    assert rc == 0, err
    create = [c for c in client.calls if c[0] == "create_pod"][0][1]
    assert create["gpuTypeIds"] == ["NVIDIA H200"] and create["gpuCount"] == 2
    assert create["networkVolumeId"] == "cdv10pb3cq" and create["dataCenterIds"] == ["EUR-IS-3"]
    st = read_state(env["home"], "exp1")
    assert st["id"] == "newpod789ghi" and st["status"] == "running"
    assert st["ip"] == "198.51.100.7" and st["port"] == 22222
    assert st["volume"] == "cdv10pb3cq" and st["volume_dc"] == "EUR-IS-3"
    assert st["keys"] == ["laptop@test", "runpodctl"]
    assert (env["home"] / "pods" / "exp1.known_hosts").exists()
    # ssh liveness probe used the resolved ip/port and the per-pod known_hosts
    probe = fake_ssh.calls[-1]["argv"]
    assert "root@198.51.100.7" in probe and probe[probe.index("-p") + 1] == "22222"
    assert f"UserKnownHostsFile={env['home'] / 'pods' / 'exp1.known_hosts'}" in probe
    assert "pod ready" in out and "rp bootstrap exp1" in out


def test_up_by_datacenter_and_ambiguity(env, client, fake_ssh, capsys):
    rc, out, err = run_cli(["up", "--name", "e2", "--volume", "US-MD-1"], capsys)
    assert rc == 0, err
    assert [c for c in client.calls if c[0] == "create_pod"][0][1]["networkVolumeId"] == "9hcgyw68jr"
    rc, out, err = run_cli(["up", "--name", "e3", "--volume", "EUR-IS-3"], capsys)
    assert rc == 1 and "2 network volumes in EUR-IS-3" in err and "cdv10pb3cq" in err
    rc, out, err = run_cli(["up", "--name", "e4", "--volume", "nosuchvol"], capsys)
    assert rc == 1 and "not found" in err


def test_up_name_collision(env, client, fake_ssh, capsys):
    assert run_cli(["up", "--name", "dup"], capsys)[0] == 0
    rc, out, err = run_cli(["up", "--name", "dup"], capsys)
    assert rc == 1 and "already exists" in err
    assert sum(1 for c in client.calls if c[0] == "create_pod") == 1


def test_up_no_wait(env, client, fake_ssh, capsys):
    rc, out, err = run_cli(["up", "--name", "nw", "--no-wait"], capsys)
    assert rc == 0 and out.strip() == "newpod789ghi"
    assert read_state(env["home"], "nw")["status"] == "creating"
    assert not fake_ssh.calls


def test_up_api_error_is_loud(env, client, fake_ssh, capsys, monkeypatch):
    from rp.api import APIError

    def boom(body):
        raise APIError("POST", "/pods", 400, '{"error":"There are no longer any instances available"}')

    monkeypatch.setattr(client, "create_pod", boom)
    rc, out, err = run_cli(["up", "--name", "fail"], capsys)
    assert rc == 1 and "HTTP 400" in err and "no longer any instances" in err


# -- ls / status / volumes ---------------------------------------------------------------

def test_ls_table_and_json(env, client, capsys):
    rc, out, err = run_cli(["ls"], capsys)
    assert rc == 0
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["NAME", "ID", "STATUS"]
    assert "demo" in lines[1] and "RUNNING" in lines[1] and "203.0.113.10:10406" in lines[1] and "3.29" in lines[1]
    assert "old-one" in lines[2] and "stopped" in lines[2] and "NVIDIA H200 x2" in lines[2]
    assert "1 running, $3.29/h total" in out
    rc, out, err = run_cli(["ls", "--json"], capsys)
    data = json.loads(out)
    assert {p["id"] for p in data} == {"podrun123abc", "podstop456def"}
    assert all("env" not in p for p in data)  # redacted


def test_status_by_name_and_id(env, client, capsys):
    run_cli(["adopt", "podrun123abc", "--name", "mine"], capsys)
    rc, out, err = run_cli(["status", "mine"], capsys)
    assert rc == 0 and "ssh root@203.0.113.10 -p 10406" in out and "(local: mine)" in out
    rc, out, err = run_cli(["status", "podstop456def", "--json"], capsys)
    d = json.loads(out)
    assert d["state"] is None and d["pod"]["id"] == "podstop456def" and "env" not in d["pod"]
    rc, out, err = run_cli(["status", "nosuch"], capsys)
    assert rc == 1 and "404" in err


def test_volumes(env, client, capsys):
    rc, out, err = run_cli(["volumes"], capsys)
    assert rc == 0 and "cdv10pb3cq" in out and "EUR-IS-3" in out and "800GB" in out
    rc, out, err = run_cli(["volumes", "--json"], capsys)
    assert len(json.loads(out)) == 3


# -- ssh / scp -----------------------------------------------------------------------

def _adopt(capsys, name="mine"):
    rc, out, err = run_cli(["adopt", "podrun123abc", "--name", name], capsys)
    assert rc == 0, err


def test_ssh_command_and_interactive(env, client, fake_ssh, capsys):
    _adopt(capsys)
    rc, out, err = run_cli(["ssh", "mine", "--", "nvidia-smi", "&&", "df", "-h"], capsys)
    assert rc == 0
    call = fake_ssh.calls[-1]
    assert call["remote"] == "nvidia-smi && df -h"
    assert "root@203.0.113.10" in call["argv"] and "10406" in call["argv"]
    rc, _, _ = run_cli(["ssh", "mine"], capsys)  # interactive -> subprocess.call
    assert rc == 0 and "BatchMode=yes" not in fake_ssh.calls[-1]["argv"]


def test_ssh_remote_exit_code_propagates(env, client, fake_ssh, capsys):
    _adopt(capsys)
    fake_ssh.add("false", returncode=7)
    assert run_cli(["ssh", "mine", "--", "false"], capsys)[0] == 7


def test_scp_prefix(env, client, fake_ssh, capsys):
    _adopt(capsys)
    rc, out, err = run_cli(["scp", "mine", ".env", "pod:/workspace/repo/.env"], capsys)
    assert rc == 0, err
    argv = fake_ssh.calls[-1]["argv"]
    assert argv[0] == "scp" and argv[-2:] == [".env", "root@203.0.113.10:/workspace/repo/.env"]
    rc, out, err = run_cli(["scp", "mine", "pod:/workspace/out.json", "./", "-r"], capsys)
    argv = fake_ssh.calls[-1]["argv"]
    assert argv[-2:] == ["root@203.0.113.10:/workspace/out.json", "./"] and "-r" in argv
    rc, out, err = run_cli(["scp", "mine", "a", "b"], capsys)
    assert rc == 1 and "pod:" in err


# -- bootstrap ----------------------------------------------------------------------------

def test_bootstrap_uploads_and_runs(env, client, fake_ssh, capsys, tmp_path):
    _adopt(capsys)
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=sk-fake\n")
    rc, out, err = run_cli([
        "bootstrap", "mine", "--repo", "https://github.com/timf34/AttractorBench.git", "--branch", "main",
        "--env", str(dotenv), "--profile", "hf-latest", "--req", "requirements-extra.txt",
    ], capsys)
    assert rc == 0, err
    scps = [c["argv"] for c in fake_ssh.calls if c["argv"][0] == "scp"]
    dsts = [a[-1] for a in scps]
    assert "root@203.0.113.10:/workspace/.rp/bootstrap.sh" in dsts
    assert "root@203.0.113.10:/workspace/.rp/req/hf-latest.txt" in dsts
    assert "root@203.0.113.10:/workspace/.rp/dotenv" in dsts
    run = fake_ssh.remote_cmds()[-1]
    assert run.endswith("bash /workspace/.rp/bootstrap.sh")
    assert "RP_REPO_URL=https://github.com/timf34/AttractorBench.git" in run
    assert "RP_REPO_DIR=/workspace/AttractorBench" in run
    assert "RP_VENV=/workspace/AttractorBench_venv" in run
    assert "RP_BRANCH=main" in run
    assert "RP_MIN_CUDA=12.8" in run
    assert "RP_REQS='/workspace/.rp/req/hf-latest.txt /workspace/AttractorBench/requirements-extra.txt'" in run
    assert "RP_DOTENV=/workspace/.rp/dotenv" in run
    assert "RP_GIT_SSH_COMMAND=''" in run
    st = read_state(env["home"], "mine")
    assert st["repo_dir"] == "/workspace/AttractorBench" and st["venv"] == "/workspace/AttractorBench_venv"
    # second run without --repo reuses state and defaults to the repo's requirements.txt
    rc, out, err = run_cli(["bootstrap", "mine"], capsys)
    assert rc == 0, err
    run2 = fake_ssh.remote_cmds()[-1]
    assert "RP_REQS=''" in run2 and "RP_REQS_OPT=/workspace/AttractorBench/requirements.txt" in run2


def test_bootstrap_failure_is_loud(env, client, fake_ssh, capsys):
    _adopt(capsys)
    fake_ssh.add("bash /workspace/.rp/bootstrap.sh", returncode=3)
    rc, out, err = run_cli(["bootstrap", "mine", "--repo", "https://github.com/o/r"], capsys)
    assert rc == 1 and "exit 3" in err and "bootstrap.log" in err


def test_bootstrap_deploy_key_flow(env, client, fake_ssh, capsys, monkeypatch):
    _adopt(capsys)
    fake_ssh.add("deploy_ed25519.pub", stdout="ssh-ed25519 AAAAC3PODKEY rp-mine\n")
    gh_calls = []

    def fake_gh(*args):
        gh_calls.append(args)
        if "-X" in args and "DELETE" in args:
            return ""
        return json.dumps({"id": 160700000, "key": "ssh-ed25519 AAAAC3PODKEY", "read_only": False})

    monkeypatch.setattr(rp.bootstrap, "_gh", fake_gh)
    rc, out, err = run_cli(["bootstrap", "mine", "--repo", "https://github.com/timf34/AttractorBench.git", "--deploy-key"], capsys)
    assert rc == 0, err
    assert gh_calls[0][:2] == ("api", "repos/timf34/AttractorBench/keys")
    assert "key=ssh-ed25519 AAAAC3PODKEY rp-mine" in gh_calls[0] and "read_only=false" in gh_calls[0]
    keygen = [c for c in fake_ssh.remote_cmds() if "ssh-keygen" in c][0]
    assert "-t ed25519" in keygen and "ssh-keyscan -t ed25519 github.com" in keygen
    run = fake_ssh.remote_cmds()[-1]
    assert "RP_REPO_URL=git@github.com:timf34/AttractorBench.git" in run
    assert "RP_GIT_SSH_COMMAND='ssh -i /root/.ssh/deploy_ed25519 -o IdentitiesOnly=yes'" in run
    st = read_state(env["home"], "mine")
    assert st["deploy_key"]["id"] == 160700000 and st["deploy_key"]["repo"] == "timf34/AttractorBench"
    # idempotent: same pub key -> no second registration
    rc, out, err = run_cli(["bootstrap", "mine", "--deploy-key"], capsys)
    assert rc == 0 and len([g for g in gh_calls if "-X" not in g]) == 1
    # rp down deletes the key
    rc, out, err = run_cli(["down", "mine", "--yes"], capsys)
    assert rc == 0, err
    assert ("api", "-X", "DELETE", "repos/timf34/AttractorBench/keys/160700000") in gh_calls
    assert read_state(env["home"], "mine")["deploy_key"] is None


# -- run / logs / jobs / kill --------------------------------------------------------------

def test_run_logs_jobs_kill(env, client, fake_ssh, capsys):
    _adopt(capsys)
    # pretend a bootstrap happened
    rc, _, err = run_cli(["bootstrap", "mine", "--repo", "https://github.com/o/Repo.git", "--no-req"], capsys)
    assert rc == 0, err
    fake_ssh.add("tmux new-session", stdout="started\n")
    rc, out, err = run_cli(["run", "mine", "--job", "sweep1", "--env", "K=v", "--", "python", "-u", "run.py", "--n", "3"], capsys)
    assert rc == 0, err
    call = [c for c in fake_ssh.calls if "tmux new-session" in c["remote"]][-1]
    assert "tmux new-session -d -s sweep1 'bash /workspace/.rp/jobs/sweep1.sh > /workspace/sweep1.log 2>&1; echo EXIT=$? >> /workspace/sweep1.log'" in call["remote"]
    script = call["input"]
    assert "cd /workspace/Repo ||" in script and "source /workspace/Repo_venv/bin/activate" in script
    assert "export K=v" in script and script.rstrip().endswith("python -u run.py --n 3")
    assert "rp logs mine --job sweep1 -f" in out
    st = read_state(env["home"], "mine")
    assert st["last_job"] == "sweep1" and st["jobs"]["sweep1"]["cmd"] == "python -u run.py --n 3"
    # logs defaults to last job
    rc, out, err = run_cli(["logs", "mine", "-n", "20", "-f"], capsys)
    assert "tail -n 20 -F /workspace/sweep1.log" in fake_ssh.remote_cmds()[-1]
    # jobs
    fake_ssh.add("## tmux sessions", stdout="## tmux sessions\nsweep1: 1 windows\n## jobs\nsweep1 running [rp] cmd: python\n")
    rc, out, err = run_cli(["jobs", "mine"], capsys)
    assert rc == 0 and "running" in out and "sweep1" in out and "python -u run.py" in out
    # kill
    rc, out, err = run_cli(["kill", "mine", "--job", "sweep1"], capsys)
    assert rc == 0 and "kill-session -t =sweep1" in fake_ssh.remote_cmds()[-1]


def test_run_requires_command_and_valid_job(env, client, fake_ssh, capsys):
    _adopt(capsys)
    rc, out, err = run_cli(["run", "mine", "--job", "j"], capsys)
    assert rc == 1 and "after `--`" in err
    rc, out, err = run_cli(["run", "mine", "--job", "bad.name", "--", "true"], capsys)
    assert rc == 1 and "tmux" in err
    fake_ssh.add("tmux new-session", returncode=95, stderr="job j already has a live tmux session")
    rc, out, err = run_cli(["run", "mine", "--job", "j", "--", "true"], capsys)
    assert rc == 1 and "already has a live tmux session" in err


# -- down / start / forget -----------------------------------------------------------------

def test_down_stop_and_terminate(env, client, fake_ssh, capsys):
    _adopt(capsys)
    # adopted pod (not created by rp up) + non-interactive stdin -> refuses without --yes
    rc, out, err = run_cli(["down", "mine"], capsys)
    assert rc == 1 and "adopted" in err and "--yes" in err and ("stop_pod", "podrun123abc") not in client.calls
    rc, out, err = run_cli(["down", "mine", "--yes"], capsys)
    assert rc == 0 and ("stop_pod", "podrun123abc") in client.calls
    st = read_state(env["home"], "mine")
    assert st["status"] == "stopped" and st["ip"] is None
    rc, out, err = run_cli(["down", "mine", "--terminate", "-y"], capsys)
    assert rc == 0 and ("delete_pod", "podrun123abc") in client.calls
    assert read_state(env["home"], "mine")["status"] == "terminated"
    # state is kept; forget removes it; up can reuse the name after termination
    assert (env["home"] / "pods" / "mine.json").exists()
    rc, out, err = run_cli(["forget", "mine"], capsys)
    assert rc == 0 and not (env["home"] / "pods" / "mine.json").exists()


def test_forget_refuses_running_without_force(env, client, capsys):
    _adopt(capsys)
    rc, out, err = run_cli(["forget", "mine"], capsys)
    assert rc == 1 and "--force" in err
    assert run_cli(["forget", "mine", "--force"], capsys)[0] == 0


def test_start_resumes_and_reresolves(env, client, fake_ssh, capsys):
    _adopt(capsys)
    assert run_cli(["down", "mine", "--yes"], capsys)[0] == 0
    kh = env["home"] / "pods" / "mine.known_hosts"
    kh.write_text("old host key\n")
    rc, out, err = run_cli(["start", "mine"], capsys)
    assert rc == 0, err
    assert ("start_pod", "podrun123abc") in client.calls
    st = read_state(env["home"], "mine")
    assert st["status"] == "running" and st["ip"] == "198.51.100.7" and st["port"] == 22222
    assert kh.read_text() == ""  # truncated: new container, new host key


def test_unknown_pod_name_message(env, client, capsys):
    rc, out, err = run_cli(["ssh", "ghost"], capsys)
    assert rc == 1 and "no local pod named 'ghost'" in err and "rp adopt" in err


def test_down_on_rp_created_pod_stops_without_prompt(env, client, fake_ssh, capsys):
    assert run_cli(["up", "--name", "own"], capsys)[0] == 0
    rc, out, err = run_cli(["down", "own"], capsys)
    assert rc == 0 and ("stop_pod", "newpod789ghi") in client.calls
    rc, out, err = run_cli(["down", "own", "--terminate"], capsys)  # non-tty: needs -y
    assert rc == 1 and "--yes" in err and ("delete_pod", "newpod789ghi") not in client.calls
    assert run_cli(["down", "own", "--terminate", "-y"], capsys)[0] == 0
    assert ("delete_pod", "newpod789ghi") in client.calls
