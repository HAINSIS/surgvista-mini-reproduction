"""Check that demo_official.disentangle equals the official ivtmetrics Disentangle (needs the official clone).

    python 05_demo/check_disentangle.py      -> prints EQUAL (instrument, verb, target) or the first difference
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(HERE))
import demo_official as D  # noqa: E402
from triplet_metrics import Disentangle  # noqa: E402  (imports ivtmetrics from <work_root>/ref/SurgVISTA)

off = Disentangle()
rng = np.random.default_rng(0)
probs = np.concatenate([rng.random((2000, 100)), (rng.random((500, 100)) > 0.9).astype(float)])
for comp in ("i", "v", "t"):
    a = np.stack([D.disentangle(p, comp) for p in probs])
    b = off.extract(probs, comp)
    assert a.shape == b.shape and np.array_equal(a, b), f"{comp}: shapes {a.shape} {b.shape}, max diff {np.abs(a - b).max()}"
print("EQUAL (instrument, verb, target) on", len(probs), "random score vectors")
