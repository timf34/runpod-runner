"""Shared fixtures: isolated RP_HOME/config, a fake RunPod client, a recorded fake ssh/scp."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

import rp.pods
import rp.ssh
import rp.api
from rp.api import APIError

POD_RUNNING = {
    "id": "podrun123abc",
    "name": "demo",
    "desiredStatus": "RUNNING",
    "costPerHr": 3.29,
    "gpuCount": 1,
    "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "containerDiskInGb": 60,
    "publicIp": "203.0.113.10",
    "portMappings": {"22": 10406},
    "ports": ["22/tcp", "8888/http"],
    "env": {"PUBLIC_KEY": "ssh-ed25519 AAA x", "JUPYTER_PASSWORD": "secret"},
    "networkVolumeId": "cdv10pb3cq",
    "networkVolume": {"id": "cdv10pb3cq", "name": "regional_indigo_ocelot_volume", "dataCenterId": "EUR-IS-3", "size": 800},
    "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3", "dataCenterId": "EUR-IS-3"},
    "createdAt": "2026-08-19 13:15:43.832 +0000 UTC",
    "lastStatusChange": "Rented by User: Wed Aug 19 2026",
    "volumeMountPath": "/workspace",
    "volumeInGb": 0,
}
POD_STOPPED = {
    "id": "podstop456def",
    "name": "old-one",
    "desiredStatus": "EXITED",
    "costPerHr": 2.29,
    "gpuCount": 2,
    "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "publicIp": "",
    "portMappings": None,
    "env": {},
    "machine": {"gpuTypeId": "NVIDIA H200", "dataCenterId": "AP-JP-1"},
    "createdAt": "2026-08-17 20:14:44.973 +0000 UTC",
}
VOLUMES = [
    {"id": "cdv10pb3cq", "name": "regional_indigo_ocelot_volume", "dataCenterId": "EUR-IS-3", "size": 800},
    {"id": "fgb3c4kqo4", "name": "witty_jade_toucan_volume", "dataCenterId": "EUR-IS-3", "size": 800},
    {"id": "9hcgyw68jr", "name": "qualified_brown_squirrel_volume", "dataCenterId": "US-MD-1", "size": 800},
]


class FakeClient:
    """Stands in for RunPodClient. Records calls; `create_pod` returns a pod that gets its ip on the 2nd GET."""

    def __init__(self):
        self.pods = {POD_RUNNING["id"]: copy.deepcopy(POD_RUNNING), POD_STOPPED["id"]: copy.deepcopy(POD_STOPPED)}
        self.volumes = copy.deepcopy(VOLUMES)
        self.calls: list[tuple] = []
        self.gets_until_ip = 2  # how many GETs before a freshly created pod reports ip/port
        self._gets: dict[str, int] = {}

    def list_pods(self):
        self.calls.append(("list_pods",))
        return [copy.deepcopy(p) for p in self.pods.values()]

    def get_pod(self, pod_id):
        self.calls.append(("get_pod", pod_id))
        if pod_id not in self.pods:
            raise APIError("GET", f"/pods/{pod_id}", 404, '{"error":"pod not found"}')
        self._gets[pod_id] = self._gets.get(pod_id, 0) + 1
        pod = self.pods[pod_id]
        if pod.get("_pending") and self._gets[pod_id] >= self.gets_until_ip:
            pod.pop("_pending")
            pod["publicIp"] = "198.51.100.7"
            pod["portMappings"] = {"22": 22222}
        return copy.deepcopy(pod)

    def create_pod(self, body):
        self.calls.append(("create_pod", copy.deepcopy(body)))
        pod = {"id": "newpod789ghi", "name": body["name"], "desiredStatus": "RUNNING", "costPerHr": 2.5,
               "publicIp": "", "portMappings": None, "_pending": True, "gpuCount": body.get("gpuCount", 1),
               "imageName": body.get("imageName"), "machine": {"gpuTypeId": body["gpuTypeIds"][0], "dataCenterId": "EUR-IS-3"},
               "createdAt": "2026-08-19 14:00:00.000 +0000 UTC", "env": body.get("env", {})}
        self.pods[pod["id"]] = pod
        return copy.deepcopy(pod)

    def stop_pod(self, pod_id):
        self.calls.append(("stop_pod", pod_id))
        self.pods[pod_id]["desiredStatus"] = "EXITED"
        self.pods[pod_id]["publicIp"] = ""
        return {"id": pod_id, "desiredStatus": "EXITED"}

    def start_pod(self, pod_id):
        self.calls.append(("start_pod", pod_id))
        p = self.pods[pod_id]
        p["desiredStatus"] = "RUNNING"
        p["_pending"] = True
        self._gets[pod_id] = 0
        return {"id": pod_id}

    def delete_pod(self, pod_id):
        self.calls.append(("delete_pod", pod_id))
        self.pods.pop(pod_id, None)

    def list_volumes(self):
        self.calls.append(("list_volumes",))
        return copy.deepcopy(self.volumes)


class FakeProc:
    """A scripted subprocess.run replacement for ssh/scp. `script` maps substrings of the remote
    command to (returncode, stdout, stderr); default is success with empty output."""

    def __init__(self):
        self.calls: list[dict] = []
        self.script: list[tuple[str, int, str, str]] = []
        self.default = (0, "", "")

    def add(self, needle: str, returncode=0, stdout="", stderr=""):
        self.script.append((needle, returncode, stdout, stderr))

    def __call__(self, argv, **kw):
        remote = argv[-1] if argv and argv[0] == "ssh" else ""
        self.calls.append({"argv": list(argv), "remote": remote, "input": kw.get("input"), "kw": kw})
        rc, out, err = self.default
        for needle, r, o, e in self.script:
            if needle in " ".join(argv):
                rc, out, err = r, o, e
                break
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)

    def remote_cmds(self) -> list[str]:
        return [c["remote"] for c in self.calls if c["argv"][0] == "ssh"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated RP_HOME + config with a fake ssh key; RUNPOD_API_KEY set to a dummy."""
    home = tmp_path / "rp_home"
    monkeypatch.setenv("RP_HOME", str(home))
    key = tmp_path / "id_ed25519"
    key.write_text("fake private key\n")
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAAC3FAKEKEY laptop@test\n")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[defaults]\nssh_key = "{key}"\nextra_public_keys = ["ssh-rsa AAAAB3FAKERSA runpodctl"]\n')
    os.chmod(cfg, 0o600)
    monkeypatch.setenv("RP_CONFIG", str(cfg))
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-not-real")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(rp.pods.time, "sleep", lambda s: None)
    monkeypatch.setattr(rp.ssh.time, "sleep", lambda s: None)
    return {"home": home, "key": key, "config": cfg, "tmp": tmp_path}


@pytest.fixture
def client(env, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(rp.pods, "get_client", lambda cfg: fake)
    return fake


@pytest.fixture
def fake_ssh(env, monkeypatch):
    fp = FakeProc()
    monkeypatch.setattr(rp.ssh.subprocess, "run", fp)
    monkeypatch.setattr(rp.ssh.subprocess, "call", lambda argv, **kw: (fp(argv, **kw).returncode))
    return fp


def run_cli(argv: list[str], capsys=None) -> tuple[int, str, str]:
    from rp.cli import main
    rc = main(argv)
    if capsys is None:
        return rc, "", ""
    out = capsys.readouterr()
    return rc, out.out, out.err


def read_state(home: Path, name: str) -> dict:
    return json.loads((home / "pods" / f"{name}.json").read_text())
