"""Environment fingerprint stamped into every run.json (bench-spec.md §6).

Best-effort by design: a missing git binary or torch import failure turns
into a null field, never a failed run — the fingerprint exists to make
history comparable, not to gate execution.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys


def _git(args: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def fingerprint(repo_dir: str | None = None) -> dict:
    sha = _git(["rev-parse", "HEAD"], cwd=repo_dir)
    status = _git(["status", "--porcelain"], cwd=repo_dir)

    torch_version = None
    device_name = None
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "Apple MPS"
    except Exception:
        pass

    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch_version,
        "device_name": device_name,
        "cpu_count": os.cpu_count(),
    }
