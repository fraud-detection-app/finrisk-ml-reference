"""Shared setup: make finrisk_jobs/, pipeline/ and tests/ importable, find the app's source, and give every test a fake AWS."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "pipeline", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def _find_app_src() -> Path | None:
    for candidate in (os.environ.get("FINRISK_APP_SRC"), ROOT.parent / "finrisk-app" / "src"):
        if candidate and (Path(candidate) / "financial_risk").is_dir():
            return Path(candidate).resolve()
    return None


APP_SRC = _find_app_src()
if APP_SRC:
    os.environ["FINRISK_APP_SRC"] = str(APP_SRC)          # the jobs find the app's package the same way the tests do
    if str(APP_SRC) not in sys.path:
        sys.path.insert(0, str(APP_SRC))

from fakes import World  # noqa: E402

needs_app = pytest.mark.skipif(APP_SRC is None, reason="finrisk-app's source not found: set FINRISK_APP_SRC=../finrisk-app/src")

# Folders that tools generate (or that hold git's own bookkeeping). A scan of "what is committed" must never read them:
# on a CI runner .git/ holds fetched refs and shas; .venv/ holds other people's code.
GENERATED_DIRS = {".git", ".terraform", ".venv", "venv", "__pycache__", ".pytest_cache", "output", "model"}


def repo_files(root: Path = ROOT) -> list[Path]:
    """The files of the repository that git tracks (or would track): `git ls-files` when `root` is a git checkout and git
    is installed, otherwise a walk of the folder that does not enter GENERATED_DIRS or another
    checkout nested inside it (CI may check a sibling repository out into a subfolder). Either way nothing from a generated
    folder, and never a .pyc."""
    names: list[Path] | None = None
    if (root / ".git").exists() and shutil.which("git"):
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                              capture_output=True, text=True)
        if done.returncode == 0:
            names = [root / n for n in done.stdout.split("\0") if n]
    if names is None:
        names = []
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in GENERATED_DIRS and not (Path(folder) / d / ".git").exists()]            # prune: do not even descend
            names.extend(Path(folder) / f for f in files)
    return sorted(p for p in names if p.is_file() and p.suffix != ".pyc"
                  and not GENERATED_DIRS.intersection(p.relative_to(root).parts))


@pytest.fixture
def world(capsys):
    return World(capsys)


@pytest.fixture(autouse=True)
def clean_job_clients():
    """The jobs keep their AWS clients in a module-level registry; every test starts with none."""
    from finrisk_jobs import common

    common.reset_clients()
    yield
    common.reset_clients()
