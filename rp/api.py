"""Thin client for the RunPod REST API (https://rest.runpod.io/v1).

Only the handful of endpoints rp needs. Errors carry the HTTP status and the response
body so nothing fails silently.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from . import RpError, __version__

BASE_URL = "https://rest.runpod.io/v1"
RETRY_STATUSES = (502, 503, 504)


class APIError(RpError):
    def __init__(self, method: str, path: str, status: int, body: str):
        self.method, self.path, self.status, self.body = method, path, status, body
        super().__init__(f"RunPod API {method} {path} -> HTTP {status}: {_short(body)}")


def _short(body: str, n: int = 800) -> str:
    body = (body or "").strip() or "(empty body)"
    return body if len(body) <= n else body[:n] + f"... ({len(body)} bytes)"


class RunPodClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout: float = 60,
        session: requests.Session | None = None,
        retries: int = 3,
    ):
        if not api_key:
            raise RpError("RunPodClient needs an API key")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"runpod-runner/{__version__}",
            }
        )

    # -- core -------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        expect: tuple[int, ...] = (200,),
    ) -> Any:
        url = self.base_url + path
        attempt = 0
        while True:
            attempt += 1
            try:
                r = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout)
            except requests.RequestException as e:
                if method == "GET" and attempt < self.retries:
                    time.sleep(2 * attempt)
                    continue
                raise RpError(f"RunPod API {method} {path}: connection error: {e}") from e
            if r.status_code in RETRY_STATUSES and method == "GET" and attempt < self.retries:
                time.sleep(2 * attempt)
                continue
            if r.status_code not in expect:
                raise APIError(method, path, r.status_code, r.text)
            if r.status_code == 204 or not (r.content or b"").strip():
                return None
            try:
                return r.json()
            except ValueError as e:
                raise RpError(f"RunPod API {method} {path}: non-JSON response: {_short(r.text, 300)}") from e

    # -- pods -------------------------------------------------------------
    _POD_PARAMS = {"includeMachine": "true", "includeNetworkVolume": "true"}

    def list_pods(self) -> list[dict]:
        data = self._request("GET", "/pods", params=dict(self._POD_PARAMS))
        return list(data or [])

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}", params=dict(self._POD_PARAMS))

    def create_pod(self, body: dict) -> dict:
        return self._request("POST", "/pods", json_body=body, expect=(200, 201))

    def stop_pod(self, pod_id: str) -> Any:
        return self._request("POST", f"/pods/{pod_id}/stop", expect=(200, 201, 204))

    def start_pod(self, pod_id: str) -> Any:
        return self._request("POST", f"/pods/{pod_id}/start", expect=(200, 201, 204))

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}", expect=(200, 204))

    # -- volumes ----------------------------------------------------------
    def list_volumes(self) -> list[dict]:
        data = self._request("GET", "/networkvolumes")
        return list(data or [])
