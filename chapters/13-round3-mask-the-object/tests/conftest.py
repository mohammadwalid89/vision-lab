"""Test setup.

Parity tests compare the library with the R1/R2 scorer, gate and ordering modules.
Set R1_SCORER_DIR=src/vision_lab/wsp/legacy to run the scorer and gate parity tests (64 pass).
The ordering parity test needs ordered.py from the Round 1 run, which is not published, and
stays skipped; without R1_SCORER_DIR all parity tests are skipped.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from vision_lab.wsp.runtime import limit

sys.dont_write_bytecode = True
limit()

_dir = os.environ.get("R1_SCORER_DIR")
R1_COPY = Path(_dir) if _dir else None


def _load(name):
    if R1_COPY is None or not (R1_COPY / f"{name}.py").exists():
        pytest.skip("set R1_SCORER_DIR to run the R1 parity tests")
    spec = importlib.util.spec_from_file_location(f"r1copy_{name}", R1_COPY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def r1_scorer():
    return _load("scorer")


@pytest.fixture(scope="session")
def r1_bars():
    return _load("dev_bars")


@pytest.fixture(scope="session")
def r1_ordered():
    return _load("ordered")
