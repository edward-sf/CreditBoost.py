"""Content hashing. Deliberately dependency-free.

This lives apart from data.py because data.py imports scikit-learn, which is
not installed in the Docker builder stage. The artifact CLI runs there and
needs to hash files, so the function cannot live behind an sklearn import.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    """Content hash, used both to trace a model to its training data and to
    verify a downloaded release asset against the lockfile."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
