"""Importing a backend must not keep the importer's frames alive (a stored exception's traceback
did: the first caller's arrays and models stayed referenced for the life of the process)."""
import subprocess
import sys

import pytest

pytest.importorskip("torch")

SCRIPT = r"""
import gc, sys, weakref
sys.modules["triton"] = None                  # as on a box without triton: the import fails

class Big: pass

def first_call():
    big = Big()
    import rankfield.backends.triton_gpu as t
    assert not t.available() and "triton" in t.why_unavailable()
    return weakref.ref(big)

ref = first_call()
gc.collect()
print("freed" if ref() is None else "leaked")
"""


def test_a_failed_triton_import_holds_no_frames():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "freed", out.stdout + out.stderr
