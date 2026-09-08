"""Which code produced an artifact — captured at build time, never inferred later.

Why this exists
---------------
``feature_schema_version`` tracks the NAMES a model consumes. It does not track
what those names MEAN. When a leakage bug in ``feature_builder`` is fixed, the
feature keys are unchanged, the schema version is unchanged, and two models — one
trained on leaky values, one on corrected ones — carry byte-identical metadata.
Nothing distinguishes them.

A commit hash does. It is the only field that answers "what code was this built
with", and it cannot be reconstructed afterwards: ``git log --before <date>`` finds
the commit that existed, not the commit that ran, and the difference is exactly a
dirty working tree.

Why ``dirty`` matters more than the hash
----------------------------------------
A hash recorded from a dirty tree names code that IS NOT WHAT RAN. Recording it
alone is provenance theatre — it looks reproducible and is not. ``dirty`` does not
fix that; it states it, which is the honest outcome and lets a reader downgrade
their confidence instead of misplacing it.

Untracked files count as dirty. An untracked ``.py`` inside the package is imported
and executed like any other module, so ``--untracked-files=no`` would report clean
while unversioned code was running.

Never raises
------------
Provenance is metadata. A missing ``git`` binary, a source tree exported without
``.git/`` (the Docker image is exactly this — ``.dockerignore`` excludes it), or a
hung subprocess must not abort a training run that took twenty minutes of CPU. Every
failure degrades to ``(None, None)``, which reads as "not recorded" — distinct from
``dirty=False``, which is a positive claim that the tree was clean.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# app/services/provenance.py -> app/services -> app -> backend -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class GitProvenance:
    """``commit``/``dirty`` are both ``None`` when git could not be consulted.

    ``dirty is None`` and ``dirty is False`` are different statements: the first
    means "we do not know", the second means "we checked and the tree was clean".
    Callers that collapse them lose the distinction the type exists to preserve.
    """

    commit: str | None
    dirty: bool | None

    @property
    def recorded(self) -> bool:
        return self.commit is not None


def _git(root: Path, *args: str) -> str | None:
    """Run a git command, returning stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=True,
        )
    except FileNotFoundError:
        logger.debug("git provenance: no git binary on PATH")
        return None
    except subprocess.CalledProcessError as exc:
        logger.debug("git provenance: %s failed (%s)", args, (exc.stderr or "").strip()[:120])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("git provenance: %s timed out after %ss", args, _TIMEOUT_SECONDS)
        return None
    return out.stdout.strip()


def git_provenance(root: Path | None = None) -> GitProvenance:
    """Current commit and whether the working tree has uncommitted changes.

    Args:
        root: directory to resolve the repository from. Defaults to the repo root
            rather than ``backend/`` — git resolves upward from a subdirectory, so
            either works, but naming the root makes the intent explicit.

    Returns:
        :class:`GitProvenance`. ``(None, None)`` if git could not be consulted at all.
        A commit with ``dirty=None`` is not returned: if the status probe fails after
        the rev-parse succeeded, the whole result degrades, because a hash presented
        without a cleanliness claim invites the reader to assume clean.
    """
    root = root or _REPO_ROOT
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return GitProvenance(None, None)

    # --porcelain includes untracked files by default. That is deliberate: an
    # untracked module still runs.
    status = _git(root, "status", "--porcelain")
    if status is None:
        return GitProvenance(None, None)

    return GitProvenance(commit=commit, dirty=bool(status))
