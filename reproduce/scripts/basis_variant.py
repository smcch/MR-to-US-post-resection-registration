"""Material-variant wrapper around biomech_basis.py (RESECT ablation).

Usage: basis_variant.py <variant> [biomech_basis args...]
  ratio1      no material change (identical to stock biomech_basis)
  ratio4      MAT_TUMOR = 12 kPa (4x parenchyma). Pass --tumor <seg> too:
              without a tumor seg this is identical to ratio1.
  ratioCSFx3  MAT_CSF E = 300 Pa (0.3 kPa) instead of 100 Pa.

Monkeypatches the module constants before calling biomech_basis.main();
material assignment reads the globals at call time, so this is exact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_basis as bb

variant = sys.argv[1]
if variant == "ratio4":
    bb.MAT_TUMOR = (12000.0, 0.45)
elif variant == "ratioCSFx3":
    bb.MAT_CSF = (300.0, 0.49)
elif variant != "ratio1":
    raise SystemExit(f"unknown variant {variant}")
print(f"[variant {variant}] MAT_TUMOR={bb.MAT_TUMOR} MAT_CSF={bb.MAT_CSF}")
sys.argv = [sys.argv[0]] + sys.argv[2:]
bb.main()
