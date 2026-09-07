"""The numpy path stands on its own.

torch is the ``torch`` extra, so importing ``rankfield`` and decoding a stored field must
work without it - the encoder and the restore are what need it. The check runs in a child
interpreter with torch blocked at import, because this suite has torch installed.
"""
import json
import subprocess
import sys

import numpy as np

import rankfield as rf
from conftest import logits

CHILD = r'''
import sys

class NoTorch:
    """torch is not installed, as far as this interpreter is concerned."""
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return None

sys.meta_path.insert(0, NoTorch())

import json
import numpy as np
import rankfield as rf

assert "torch" not in sys.modules, "importing rankfield pulled torch in"

d = np.load(sys.argv[1])
meta = json.loads(sys.argv[2])
code = rf.RankField(ranks=d["ranks"], support=d["support"], tail=d["tail"], meta=meta)

out = {"levels": rf.levels(meta)[:4].tolist(),
       "margin": rf.margin(code, 1).ravel()[:8].tolist(),
       "deficit": rf.deficit(code, 1).ravel()[:8].tolist()}
ids, p = rf.probabilities(code)
out["p"] = p[:, 0, 0, 0].tolist()
out["tail_at"] = int(rf.tail_at(code, 1.0).sum())
assert "torch" not in sys.modules, "a numpy decoder pulled torch in"

try:
    rf.encode(np.zeros((3, 2, 2, 2)))
except ModuleNotFoundError as e:
    out["encode_error"] = str(e)
else:
    raise AssertionError("encode() must refuse without torch")

print(json.dumps(out))
'''


def _child(tmp_path, code):
    npz = tmp_path / "field.npz"
    np.savez(npz, ranks=code.ranks, support=code.support, tail=code.tail)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    r = subprocess.run([sys.executable, str(script), str(npz), json.dumps(code.meta)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_the_numpy_decoders_read_a_field_with_torch_uninstallable(tmp_path):
    code = rf.encode(logits(K=6), depth=3)
    got = _child(tmp_path, code)
    ids, p = rf.probabilities(code)
    np.testing.assert_allclose(got["levels"], rf.levels(code.meta)[:4])
    np.testing.assert_allclose(got["margin"], rf.margin(code, 1).ravel()[:8])
    np.testing.assert_allclose(got["deficit"], rf.deficit(code, 1).ravel()[:8])
    np.testing.assert_allclose(got["p"], p[:, 0, 0, 0])
    assert got["tail_at"] == int(code.tail.sum())


def test_a_torch_path_names_the_extra_that_supplies_torch(tmp_path):
    got = _child(tmp_path, rf.encode(logits(K=6), depth=3))
    assert "rankfield[torch]" in got["encode_error"]
