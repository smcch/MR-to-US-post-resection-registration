"""Keep MCPO's displacement field for figure cases of the public sets (fer/mcpo_fields/<coh>/<case>/disp_vox.nii.gz)
by re-running the official script exactly as fer_mcpo.py does, WITHOUT touching the scored json; prints the TRE of the
re-run next to the scored one (the script is deterministic on this machine).
Usage: python fer_keep_mcpo_fields.py resect_after/Case25 bite_g4/09"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
import fer_mcpo as M                                                 # noqa

for arg in sys.argv[1:]:
    coh, c = arg.split("/")
    m = M.public_case(coh, c, keep=True)
    j = json.loads((C.FER / f"mcpo/{coh}/{c}.json").read_text())
    print(f"[keep] {coh} {c}: field kept; TRE of this run {m['tre']:.3f}, scored run {j['tre']:.3f}", flush=True)
