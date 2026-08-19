"""Pure-function tests: gpu aliases, config loading, state files, ssh argv, timestamps, bootstrap helpers, job scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rp import RpError
from rp.bootstrap import env_prefix, max_cuda, parse_github_repo, parse_min_cuda, repo_dir_name
from rp.config import load_config, require_api_key
from rp.gpus import resolve_gpu
from rp.jobs import build_job_script, kill_cmd, launch_remote_cmd, tail_cmd
from rp.pods import human_age, key_comment, looks_like_dc, parse_runpod_ts, pod_ssh_endpoint, pod_status
from rp.ssh import Conn, scp_argv, ssh_argv
from rp.state import PodState, load_state, save_state, validate_job, validate_name


# -- gpus ---------------------------------------------------------------------

def test_gpu_aliases():
    assert resolve_gpu("h100") == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe"]
    assert resolve_gpu("H100") == resolve_gpu("h100")
    assert resolve_gpu("a100") == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]
    assert resolve_gpu("h200") == ["NVIDIA H200"]


def test_gpu_exact_and_overrides():
    assert resolve_gpu("NVIDIA H100 PCIe") == ["NVIDIA H100 PCIe"]
    assert resolve_gpu("NVIDIA A40, NVIDIA L40S") == ["NVIDIA A40", "NVIDIA L40S"]
    assert resolve_gpu("h100", {"h100": ["NVIDIA H100 NVL"]}) == ["NVIDIA H100 NVL"]
    assert resolve_gpu("cheap", {"cheap": ["NVIDIA A40"]}) == ["NVIDIA A40"]
    with pytest.raises(RpError):
        resolve_gpu("")


# -- config -------------------------------------------------------------------

def test_config_env_key_wins_over_file(env, monkeypatch):
    env["config"].write_text('api_key = "from-file"\n[defaults]\ngpu = "a100"\n')
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    cfg = load_config()
    assert cfg.api_key == "from-env"
    assert cfg.gpu == "a100"
    monkeypatch.delenv("RUNPOD_API_KEY")
    assert load_config().api_key == "from-file"


def test_config_missing_key_message(env, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY")
    cfg = load_config()
    with pytest.raises(RpError, match="RUNPOD_API_KEY"):
        require_api_key(cfg)


def test_config_warns_on_loose_mode(env, capsys):
    os.chmod(env["config"], 0o644)
    load_config()
    assert "chmod 600" in capsys.readouterr().err


def test_config_public_keys(env):
    cfg = load_config()
    keys = cfg.public_keys()
    assert keys[0].startswith("ssh-ed25519 AAAAC3FAKEKEY")
    assert keys[1] == "ssh-rsa AAAAB3FAKERSA runpodctl"
    assert [key_comment(k) for k in keys] == ["laptop@test", "runpodctl"]


def test_config_bad_toml(env):
    env["config"].write_text("this is = not [toml\n")
    with pytest.raises(RpError, match="bad TOML"):
        load_config()


# -- state --------------------------------------------------------------------

def test_state_roundtrip_and_mode(env):
    st = PodState(name="p1", id="abc", ip="1.2.3.4", port=1234, gpu=["NVIDIA H200"], deploy_key={"repo": "o/r", "id": 5})
    path = save_state(st)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    back = load_state("p1")
    assert back.to_dict() == st.to_dict()
    # unknown keys in the file are tolerated (forward compat)
    d = json.loads(path.read_text())
    d["future_field"] = 1
    path.write_text(json.dumps(d))
    assert load_state("p1").id == "abc"


def test_state_missing_and_names(env):
    with pytest.raises(RpError, match="no local pod named"):
        load_state("nope")
    with pytest.raises(RpError):
        validate_name("bad name!")
    with pytest.raises(RpError):
        validate_name("")
    assert validate_name("ok-name_1.2") == "ok-name_1.2"
    with pytest.raises(RpError, match="tmux"):
        validate_job("has.dot")
    assert validate_job("job_1-x") == "job_1-x"


# -- ssh argv -----------------------------------------------------------------

def test_ssh_and_scp_argv(tmp_path):
    conn = Conn(host="203.0.113.10", port=10406, identity=tmp_path / "k", known_hosts=tmp_path / "kh")
    argv = ssh_argv(conn, "nvidia-smi")
    assert argv[0] == "ssh" and argv[-1] == "nvidia-smi" and argv[-2] == "root@203.0.113.10"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "10406"
    assert "-i" in argv and argv[argv.index("-i") + 1] == str(tmp_path / "k")
    assert "StrictHostKeyChecking=no" in argv
    assert f"UserKnownHostsFile={tmp_path / 'kh'}" in argv
    assert "BatchMode=yes" in argv
    assert "-t" not in argv
    assert "-t" in ssh_argv(conn, None, tty=True)
    assert "BatchMode=yes" not in ssh_argv(conn, None, batch=False)
    s = scp_argv(conn, "/local/f", "root@203.0.113.10:/workspace/f", recursive=True)
    assert s[0] == "scp" and "-P" in s and s[s.index("-P") + 1] == "10406" and "-r" in s
    assert s[-2:] == ["/local/f", "root@203.0.113.10:/workspace/f"]


# -- pods helpers --------------------------------------------------------------

def test_timestamps_and_age():
    ts = parse_runpod_ts("2026-08-19 13:15:43.832 +0000 UTC")
    assert ts is not None and ts.year == 2026 and ts.hour == 13
    assert parse_runpod_ts("2026-08-19T13:15:43Z") is not None
    assert parse_runpod_ts("garbage") is None
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    assert human_age(now - timedelta(minutes=49), now) == "49m"
    assert human_age(now - timedelta(hours=1, minutes=5), now) == "1h05m"
    assert human_age(now - timedelta(days=1, hours=17), now) == "1d17h"
    assert human_age(None) == "?"


def test_pod_helpers():
    assert pod_status({"desiredStatus": "RUNNING"}) == "running"
    assert pod_status({"desiredStatus": "EXITED"}) == "stopped"
    assert pod_status({}) == "unknown"
    assert pod_ssh_endpoint({"publicIp": "1.1.1.1", "portMappings": {"22": "10406"}}) == ("1.1.1.1", 10406)
    assert pod_ssh_endpoint({"publicIp": "", "portMappings": None}) == (None, None)
    assert looks_like_dc("EUR-IS-3") and looks_like_dc("US-TX-3") and not looks_like_dc("cdv10pb3cq")


# -- bootstrap helpers ---------------------------------------------------------

def test_github_repo_parsing():
    assert parse_github_repo("https://github.com/timf34/AttractorBench.git") == "timf34/AttractorBench"
    assert parse_github_repo("https://github.com/timf34/AttractorBench") == "timf34/AttractorBench"
    assert parse_github_repo("git@github.com:timf34/AttractorBench.git") == "timf34/AttractorBench"
    assert parse_github_repo("https://gitlab.com/x/y.git") is None
    assert repo_dir_name("https://github.com/timf34/AttractorBench.git") == "AttractorBench"
    assert repo_dir_name("git@github.com:o/r") == "r"


def test_min_cuda_parsing():
    assert parse_min_cuda("# rp profile\n# rp-min-cuda: 12.8\ntorch\n") == "12.8"
    assert parse_min_cuda("torch\n") is None
    assert max_cuda(["12.4", "", "12.8", "11.8"]) == "12.8"
    assert max_cuda(["", ""]) is None


def test_shipped_profiles_have_min_cuda():
    pdir = Path(__file__).resolve().parent.parent / "profiles"
    names = {p.stem for p in pdir.glob("*.txt")}
    assert {"hf-latest", "vllm-0.11"} <= names
    for p in pdir.glob("*.txt"):
        assert parse_min_cuda(p.read_text()), f"{p.name} lacks a '# rp-min-cuda:' header"
    assert (pdir / "bootstrap.sh").exists()


def test_env_prefix_quotes():
    assert env_prefix({"A": "x y", "B": "", "C": "it's"}) == "A='x y' B='' C='it'\"'\"'s'"


# -- jobs -------------------------------------------------------------------------

def test_job_script_and_launch():
    script = build_job_script("j1", "python -u train.py --n 3", cwd="/workspace/R", venv="/workspace/R_venv", dotenv=True, extra_env={"X": "a b"})
    assert "cd /workspace/R ||" in script
    assert "export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1" in script
    assert "source /workspace/R_venv/bin/activate" in script
    assert "source ./.env" in script
    assert "export X='a b'" in script
    assert script.rstrip().endswith("python -u train.py --n 3")
    launch = launch_remote_cmd("j1")
    assert "tmux new-session -d -s j1 'bash /workspace/.rp/jobs/j1.sh > /workspace/j1.log 2>&1; echo EXIT=$? >> /workspace/j1.log'" in launch
    assert "has-session -t =j1" in launch and "exit 95" in launch
    assert "tail -n 20 -F /workspace/j1.log" in tail_cmd("j1", 20, True)
    assert "-F" not in tail_cmd("j1", 20, False)
    k = kill_cmd("j1")
    assert "kill-session -t =j1" in k and "EXIT=KILLED" in k and 'kill -TERM -- -"$pid"' in k
