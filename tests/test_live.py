"""Optional read-only smoke test against the real RunPod account (GET /pods only).

Runs only when RUNPOD_API_KEY is set:  pytest -m live
"""

from __future__ import annotations

import os

import pytest

from rp.api import RunPodClient

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="RUNPOD_API_KEY not set")
def test_list_pods_live():
    client = RunPodClient(os.environ["RUNPOD_API_KEY"])
    pods = client.list_pods()
    assert isinstance(pods, list)
    for p in pods:
        assert "id" in p and "desiredStatus" in p
    vols = client.list_volumes()
    assert isinstance(vols, list)
