"""Security helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from flask import Request


def is_safe_redirect_url(target: str | None, request: Request) -> bool:
    """Return True when *target* is a same-host relative path."""
    if not target:
        return False
    if not target.startswith("/"):
        return False
    if target.startswith("//"):
        return False

    ref_url = urlparse(request.host_url)
    test_url = urlparse(target)
    return test_url.scheme in ("", ref_url.scheme) and test_url.netloc in ("", ref_url.netloc)


def resolve_allowed_path(stored_path: str, allowed_roots: list[Path]) -> Path | None:
    """Resolve *stored_path* when it lies under one of *allowed_roots*."""
    try:
        resolved = Path(stored_path).resolve(strict=True)
    except OSError:
        return None

    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue
    return None
