# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-folder digest of a Harbor task.

Same record format as Harbor's package publisher and Gym's Terminal Bench 4 loader:
SHA256 over sorted ``relative_path NUL file_sha256 LF`` records covering
``task.toml``, ``instruction.md``, ``README.md``, ``trajectory.json`` and everything
under ``environment/``, ``tests/``, ``solution/`` and ``steps/``. The digest travels in
``task_data`` under :data:`DIGEST_KEY` so the resources server can refuse a row whose
folder changed after materialization.
"""

import fnmatch
import hashlib
from pathlib import Path


DIGEST_KEY = "_ng_digest"

_SINGLE_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
_DIRECTORIES = ("environment", "tests", "solution", "steps")
_IGNORED = ("__pycache__", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")


def _ignored(relative: str) -> bool:
    parts = relative.split("/")
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in _IGNORED)


def content_hash(path: Path) -> str:
    """Digest the task folder at ``path``."""
    path = Path(path).resolve()
    files = [path / name for name in _SINGLE_FILES if (path / name).is_file()]
    for name in _DIRECTORIES:
        files.extend(p for p in (path / name).rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda p: p.relative_to(path).as_posix()):
        relative = file.relative_to(path).as_posix()
        if _ignored(relative):
            continue
        if not file.resolve().is_relative_to(path):
            raise ValueError(f"Task file escapes the task folder: {relative}")
        with file.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{relative}\0{file_hash}\n".encode())
    return digest.hexdigest()


def context_digest(path: Path) -> str:
    """Digest a Dockerfile build context (a task's ``environment/`` folder).

    SHA256 over sorted ``relative_path NUL file_sha256 NUL mode LF`` records of every regular file
    under ``path``, where ``mode`` is ``x`` for an executable file, ``-`` otherwise and ``l`` for a
    symbolic link, which contributes the text of its target instead of the file it points at. Two contexts with the same files build the same image, so the digest keys the build
    record (see ``nemo_gym.tasks.harbor.image_build``).
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Not a build context folder: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (p for p in path.rglob("*") if p.is_file() or p.is_symlink()), key=lambda p: p.relative_to(path).as_posix()
    )
    for file in files:
        relative = file.relative_to(path).as_posix()
        if _ignored(relative):
            continue
        if file.is_symlink():
            file_hash = hashlib.sha256(str(file.readlink()).encode()).hexdigest()
            mode = "l"
        else:
            with file.open("rb") as stream:
                file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            # COPY carries the executable bit into the image, so it is part of the context.
            mode = "x" if file.stat().st_mode & 0o111 else "-"
        digest.update(f"{relative}\0{file_hash}\0{mode}\n".encode())
    return digest.hexdigest()
