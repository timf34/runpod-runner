"""RunPodClient against a fake requests.Session (no network)."""

from __future__ import annotations

import json

import pytest
import requests

import rp.api
from rp import RpError
from rp.api import APIError, RunPodClient


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        if body is None:
            self.text, self.content = "", b""
        elif isinstance(body, str):
            self.text, self.content = body, body.encode()
        else:
            self.text = json.dumps(body)
            self.content = self.text.encode()

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(responses):
    s = FakeSession(responses)
    return RunPodClient("k3y", session=s, retries=3), s


def test_headers_and_list(monkeypatch):
    c, s = make([FakeResponse(200, [{"id": "a"}])])
    assert s.headers["Authorization"] == "Bearer k3y"
    assert c.list_pods() == [{"id": "a"}]
    call = s.calls[0]
    assert call["method"] == "GET" and call["url"] == "https://rest.runpod.io/v1/pods"
    assert call["params"] == {"includeMachine": "true", "includeNetworkVolume": "true"}


def test_create_pod_posts_body():
    body = {"name": "x", "gpuTypeIds": ["NVIDIA H200"]}
    c, s = make([FakeResponse(201, {"id": "new"})])
    assert c.create_pod(body) == {"id": "new"}
    assert s.calls[0]["method"] == "POST" and s.calls[0]["json"] == body


def test_api_error_has_status_and_body():
    c, s = make([FakeResponse(400, '{"error":"There are no longer any instances available"}')])
    with pytest.raises(APIError) as ei:
        c.create_pod({"name": "x"})
    assert ei.value.status == 400
    assert "HTTP 400" in str(ei.value) and "no longer any instances" in str(ei.value)
    assert "POST /pods" in str(ei.value)


def test_delete_204_returns_none_and_stop_start():
    c, s = make([FakeResponse(204), FakeResponse(200, {"id": "p"}), FakeResponse(200, {"id": "p"})])
    assert c.delete_pod("p") is None
    assert c.stop_pod("p") == {"id": "p"}
    assert c.start_pod("p") == {"id": "p"}
    assert [x["url"] for x in s.calls] == [
        "https://rest.runpod.io/v1/pods/p",
        "https://rest.runpod.io/v1/pods/p/stop",
        "https://rest.runpod.io/v1/pods/p/start",
    ]


def test_get_retries_on_503_and_connection_error(monkeypatch):
    monkeypatch.setattr(rp.api.time, "sleep", lambda s: None)
    c, s = make([FakeResponse(503, "upstream"), requests.ConnectionError("boom"), FakeResponse(200, {"id": "p"})])
    assert c.get_pod("p") == {"id": "p"}
    assert len(s.calls) == 3


def test_post_does_not_retry(monkeypatch):
    monkeypatch.setattr(rp.api.time, "sleep", lambda s: None)
    c, s = make([FakeResponse(503, "upstream")])
    with pytest.raises(APIError):
        c.create_pod({})
    assert len(s.calls) == 1


def test_non_json_body():
    c, s = make([FakeResponse(200, "<html>oops</html>")])
    with pytest.raises(RpError, match="non-JSON"):
        c.list_volumes()


def test_volumes():
    c, s = make([FakeResponse(200, [{"id": "v", "dataCenterId": "EUR-IS-3"}])])
    assert c.list_volumes()[0]["dataCenterId"] == "EUR-IS-3"
    assert s.calls[0]["url"].endswith("/networkvolumes")
