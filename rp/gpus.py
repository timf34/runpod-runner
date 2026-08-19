"""Friendly GPU names -> ordered lists of RunPod gpuTypeIds (first match wins at creation)."""

from __future__ import annotations

from . import RpError

GPU_ALIASES: dict[str, list[str]] = {
    "h100": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe"],
    "h100-sxm": ["NVIDIA H100 80GB HBM3"],
    "h100-pcie": ["NVIDIA H100 PCIe"],
    "a100": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"],
    "a100-sxm": ["NVIDIA A100-SXM4-80GB"],
    "h200": ["NVIDIA H200"],
    "a40": ["NVIDIA A40"],
    "l40s": ["NVIDIA L40S"],
    "a6000": ["NVIDIA RTX A6000"],
    "4090": ["NVIDIA GeForce RTX 4090"],
}


def resolve_gpu(spec: str, overrides: dict[str, list[str]] | None = None) -> list[str]:
    """Return the gpuTypeIds list for an alias, or pass exact ids through (comma-separated ok)."""
    aliases = dict(GPU_ALIASES)
    aliases.update({k.lower(): list(v) for k, v in (overrides or {}).items()})
    key = (spec or "").strip().lower()
    if not key:
        raise RpError("empty --gpu")
    if key in aliases:
        return list(aliases[key])
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    return ids


def alias_help(overrides: dict[str, list[str]] | None = None) -> str:
    aliases = dict(GPU_ALIASES)
    aliases.update(overrides or {})
    return "\n".join(f"  {k:10s} -> {', '.join(v)}" for k, v in aliases.items())
