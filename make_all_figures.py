"""Regenerate every figure of the paper.

    python make_all_figures.py            # all figures
    python make_all_figures.py 1 2 8     # only the listed ones

Figures 1-8 are produced from the analytical/ODE scripts in
``paper_figures/``; figures 9-11 come from ``make_figures.py`` and need the
trained checkpoints in ``checkpoints/``.

Output goes to ``figures/`` (override with FIG_OUTDIR) and the numerical
content of each figure to ``data/`` (override with DATA_OUTDIR).
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_figures"))

ANALYTIC = {
    1: "fig01_geometry",
    2: "fig02_validity_map",
    3: "fig03_sync_map",
    4: "fig04_multiregime",
    5: "fig05_phase_H0",
    6: "fig06_polarization",
    7: "fig07_spectral",
    8: "fig08_directionality",
}
CHECKPOINT_BASED = (9, 10, 11)


def main(which=None):
    which = which or sorted(ANALYTIC) + list(CHECKPOINT_BASED)
    t0 = time.time()

    for n in which:
        if n in ANALYTIC:
            print(f"[fig {n}] {ANALYTIC[n]}")
            importlib.import_module(ANALYTIC[n]).main()

    if any(n in CHECKPOINT_BASED for n in which):
        if not (ROOT / "checkpoints" / "runs.json").exists():
            print("[figs 10-12] skipped: checkpoints/runs.json not found "
                  "(run train.py and train_ur_curriculum.py first)")
        else:
            print("[figs 10-12] make_figures.py")
            import make_figures
            make_figures.main()

    print(f"\ndone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or None
    main(args)
