"""Tiny terminal helpers: colours (only when stdout is a tty and NO_COLOR unset) and tables."""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _wrap(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _enabled() else s


def green(s: str) -> str:
    return _wrap("32;1", s)


def yellow(s: str) -> str:
    return _wrap("33", s)


def red(s: str) -> str:
    return _wrap("31;1", s)


def dim(s: str) -> str:
    return _wrap("2", s)


def bold(s: str) -> str:
    return _wrap("1", s)


def table(headers: list[str], rows: list[list[str]], styles: list | None = None) -> str:
    """Left-aligned fixed-width table. ``styles`` is an optional per-row colour function."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [bold(fmt.format(*headers).rstrip())]
    for i, row in enumerate(rows):
        line = fmt.format(*[str(c) for c in row]).rstrip()
        style = styles[i] if styles else None
        lines.append(style(line) if style else line)
    return "\n".join(lines)


def info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
