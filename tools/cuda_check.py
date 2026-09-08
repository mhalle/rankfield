"""Run rankfield's test suite on a CUDA box, through Modal.

The Triton kernel is the one path no machine here can execute, and it is where a defect
survives longest: the label-table bounds check could not read a uint16 rank plane on ANY
torch backend, and the CUDA half of that went unnoticed until this was run (0.3.0). The
tests already parametrize over `conftest.devices()`, so on a GPU worker they cover CUDA
without a single CUDA-specific assertion to maintain.

The working TREE is mounted, not a released tag, so this checks the change in front of you.
Needs a Modal account (`modal token new`); it does not need modal installed in the project:

    uv run --no-project modal run tools/cuda_check.py
    uv run --no-project modal run tools/cuda_check.py --gpu A10G --tests tests/test_wide_ranks.py

One run is a couple of minutes of GPU time, most of it building the image the first time.
"""
from __future__ import annotations

import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent


def _duckn_spec() -> str:
    """The duckn source this project's own pyproject pins, as a pip requirement, so the
    worker cannot drift from what a local `uv sync` installs."""
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'duckn\s*=\s*\{\s*git\s*=\s*"([^"]+)"\s*,\s*tag\s*=\s*"([^"]+)"', text)
    if not m:
        raise SystemExit("pyproject.toml: no duckn git source found")
    return f"duckn @ git+{m.group(1)}@{m.group(2)}"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch>=2.7", "triton>=3.0", "numpy>=1.24", "pytest>=8", "zarr>=3.3",
                 _duckn_spec() if modal.is_local() else "duckn")     # the image builds locally
    .add_local_dir(str(ROOT / "src" / "rankfield"), remote_path="/root/pkg/rankfield")
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/tests")
)

app = modal.App("rankfield-cuda-check", image=image)


@app.function(gpu="A10G", timeout=1800)
def check(tests: str) -> dict:
    import subprocess
    import sys
    sys.path.insert(0, "/root/pkg")
    import torch

    import rankfield as rf
    from rankfield.backends import triton_gpu

    # plain builtins only: torch.__version__ is a str SUBCLASS, and returning it makes the
    # result undeserializable on a machine without torch - which is the machine this runs from
    out = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__),
           "rankfield": str(rf.__version__), "triton_available": bool(triton_gpu.available())}
    if not out["triton_available"]:
        out["why_unavailable"] = str(triton_gpu.why_unavailable())
    target = "/root/tests" if tests == "all" else f"/root/{tests}"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", target, "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd="/root",
                       env={"PYTHONPATH": "/root/pkg", "PATH": "/usr/bin:/bin"})
    out["returncode"] = int(r.returncode)
    out["output"] = str(r.stdout + r.stderr)[-4000:]
    return out


@app.local_entrypoint()
def main(gpu: str = "A10G", tests: str = "all"):
    d = check.remote(tests)
    for k in ("gpu", "torch", "rankfield", "triton_available", "why_unavailable", "returncode"):
        if k in d:
            print(f"{k:20} {d[k]}")
    print("\n" + d["output"])
    if d["returncode"] != 0:
        raise SystemExit(d["returncode"])
