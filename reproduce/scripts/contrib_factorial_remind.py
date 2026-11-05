"""Phase B on the ReMIND landmark cases (provisional landmarks; independent of the manuscript), from the C:
working copy (G: has disk errors).  Same deformables and same recipes as contrib_factorial.py:
  poses  nr   = the official ReMIND2Reg NiftyReg affine of the annotation study (arms/<case>/niftyreg_l2r/aff.txt)
         ours = our polished pose of the annotation study (arms/<case>/search_polish/aff.txt; all 8 accepted)
  new    f3d_mind, f3d_lncc, f3d_mind_lncc (reg_f3d -aff pose on the annotation working grid), mcpo2, mcpo6
  reused none (NR alone / pose alone), paper stage and v2 stage from the earlier ReMIND runs
Landmarks: annotation_remind/LANDMARKS (observer SANTI), every case including ReMIND-023 (since 2026-10-02).
Usage: python contrib_factorial_remind.py run | report
Output: /mnt/c/ReMIND_work/remind/B/<pose>/<deformable>/<case>.json, remind/B/REPORT.md
"""
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrib_factorial as B                                        # noqa
import l2r_f3d_cohorts as LF                                         # noqa
import v2_deform as V                                                # noqa
from annotation_qc import read_markups                               # noqa
from reg_core import resample_to_grid                                # noqa

R = Path("/mnt/c/ReMIND_work/remind")
OUT = R / "B"
CASES = ["ReMIND-003", "ReMIND-006", "ReMIND-009", "ReMIND-015", "ReMIND-021", "ReMIND-023", "ReMIND-024", "ReMIND-029",
         "ReMIND-036", "ReMIND-038", "ReMIND-042",              # added 2026-09-30
         "ReMIND-043", "ReMIND-059", "ReMIND-062", "ReMIND-064", "ReMIND-065"]   # added 2026-10-02
# 2026-10-02 user: MCPO enters the comparison ONLY as its original implementation (official script on the organisers'
# ReMIND2Reg files, mcpo_faithful_remind.py); its deformable on our/NR poses (mcpo2, mcpo6) is no longer run or reported.
NEW = ["f3d_mind", "f3d_lncc", "f3d_mind_lncc"]
ALL = ["none", "f3d_mind", "f3d_lncc", "f3d_mind_lncc", "paper", "v2"]


def paths(c):
    return R / f"data/{c}_USpreimri.nii.gz", R / f"data/{c}_T2.nii.gz", R / f"arms/{c}/US_fixed_grid.nii.gz"


def pose(p, c):
    return np.loadtxt(R / f"arms/{c}/{'nr' if p == 'nr' else 'ours'}_aff.txt")


def landmarks(c):
    L = R / "landmarks"
    # 2026-10-02: 023 is in LANDMARKS too (pilot MR + Santi's US) -> every case = observer SANTI
    mr, _ = read_markups(L / f"{c}-MR.mrk.json")
    obs = [read_markups(L / f"{c}-US-SANTI.mrk.json")[0]]
    ids = sorted(set(mr).intersection(*[set(o) for o in obs]))
    return np.array([mr[i] for i in ids]), np.mean([np.array([o[i] for i in ids]) for o in obs], axis=0)


def cell_f3d(p, c, kind):
    usp, mrp, wgp = paths(c)
    fd = Path("/tmp/cf3dr") / p / kind / c
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir(parents=True)
    np.savetxt(fd / "aff_init.txt", pose(p, c))
    t0 = time.time()
    LF.F3D = B.F3D_FLAGS[kind]
    disp = LF.f3d_case(usp, mrp, wgp, fd / "aff_init.txt", fd)
    mri, us = landmarks(c)
    u = np.asarray(nib.load(str(disp)).dataobj, np.float32)
    fold, sdlj = B.reg_of(u, np.asarray(nib.load(str(wgp)).dataobj) > 0)
    t = B.tre(us + B.SC.disp_at(disp, us), mri)
    shutil.rmtree(fd, ignore_errors=True)
    return dict(tre=t, fold=fold, sdlogj=sdlj, wall_s=time.time() - t0)


def cell_mcpo(p, hw, c):
    usp, mrp, _ = paths(c)
    usi, mri_img = nib.load(str(usp)), nib.load(str(mrp))
    aff, dims = V.box05(usi)
    A = pose(p, c)
    fx = resample_to_grid(usi, aff, dims).astype(np.float32)
    mv = resample_to_grid(mri_img, A @ aff, dims).astype(np.float32)
    pad = [(0, int(s % 2)) for s in fx.shape]
    tag = f"{p}{hw}{c.replace('-', '')}"
    ind, outd = Path(f"/tmp/cfrmcpo/{tag}"), Path(f"/tmp/cfrmcpoout/{tag}")
    shutil.rmtree(ind, ignore_errors=True)
    ind.mkdir(parents=True)
    outd.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.pad(fx, pad), aff), str(ind / "ReMIND2Reg_x0_0000.nii.gz"))
    nib.save(nib.Nifti1Image(np.pad(mv, pad), aff), str(ind / "ReMIND2Reg_x0_0001.nii.gz"))
    t0 = time.time()
    extra = [] if hw == 6 else ["--disp_hw", str(hw)]
    with open("/mnt/c/ReMIND_work/logs/contrib_B_remind_mcpo.log", "a") as fh:
        subprocess.run([B.PY, str(B.MCPO), "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"] + extra,
                       stdout=fh, stderr=subprocess.STDOUT)
    dp = outd / "disp_x0_0000_x0_0001.nii.gz"
    if not dp.exists():
        raise RuntimeError("MCPO produced no output")
    ui = nib.load(str(dp))
    u = np.asarray(ui.dataobj, np.float32)
    mri, us = landmarks(c)
    inv = np.linalg.inv(ui.affine)
    v = us @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    pred = (us + du @ ui.affine[:3, :3].T) @ A[:3, :3].T + A[:3, 3]
    fold, sdlj = B.reg_of(u, np.pad(fx, pad) > 0)
    shutil.rmtree(ind, ignore_errors=True)
    shutil.rmtree(outd, ignore_errors=True)
    return dict(tre=B.tre(pred, mri), fold=fold, sdlogj=sdlj, wall_s=time.time() - t0)


def abstained(c):
    """Gate decision of the annotation study (search rotation >= 9.5 deg -> the pipeline keeps the identity pose).
    Earlier cases were all accepted; their arms live on G: and only the new cases' arms are on C:."""
    f = Path("/mnt/c/ReMIND_work/results/annotation_remind/arms") / c / "search_source.json"
    return bool(json.loads(f.read_text())["abstain"]) if f.exists() else False


def reused():
    out = {}
    for c in CASES:
        f = R / f"reuse/{c}/TRE_sets.csv"
        prev = {r["arm"]: float(r["mean"]) for r in csv.DictReader(open(f)) if r["set"] == "consensus"}
        out[("nr", "none", c)] = prev["niftyreg_l2r"]
        out[("ours", "none", c)] = prev["initial"] if abstained(c) else prev["pose_polished"]   # abstained: our pose = identity
        out[("ours", "paper", c)] = prev["ours"]
        out[("ours", "v2", c)] = prev.get("v2", prev.get("v2_R3sym"))
    for r in csv.DictReader(open(R / "reuse/nr_affine_PER_CASE.csv")):
        out[("nr", "paper", r["case"])] = float(r["nraff_paper"])
        out[("nr", "v2", r["case"])] = float(r["nraff_v2"])
    return out


def do_run():
    for p in ("nr", "ours"):
        for d in NEW:
            for c in CASES:
                f = OUT / p / d / f"{c}.json"
                if f.exists():
                    continue
                try:
                    m = cell_f3d(p, c, d) if d.startswith("f3d") else cell_mcpo(p, int(d[-1]), c)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(m))
                    print(f"[BR] {p}/{d} {c}: TRE {m['tre']:.2f}, fold {m['fold']:.3f} %, SDLogJ {m['sdlogj']:.3f}, {m['wall_s']:.0f} s", flush=True)
                except Exception as ex:                              # noqa: BLE001
                    print(f"[BR] {p}/{d} {c} FAILED: {ex}", flush=True)
    print("REMIND B RUN DONE", flush=True)


def do_report():
    T = reused()
    for p in ("nr", "ours"):
        for d in NEW:
            for c in CASES:
                f = OUT / p / d / f"{c}.json"
                if f.exists():
                    T[(p, d, c)] = json.loads(f.read_text())["tre"]
    mf = {r["case"]: r for r in csv.DictReader(open(R / "reuse/mcpo_faithful_PER_CASE.csv"))}
    L = ["# Phase B on the ReMIND landmark cases (provisional landmarks, observer SANTI)", ""]
    for p, lab in (("nr", "official ReMIND2Reg affine (NiftyReg)"), ("ours", "our polished pose")):
        L += [f"## Pose: {lab}", "", "| case | " + " | ".join(ALL) + " |", "|---|" + "---|" * len(ALL)]
        for c in CASES:
            L.append(f"| {c} | " + " | ".join(f"{T[(p, d, c)]:.2f}" if (p, d, c) in T else "—" for d in ALL) + " |")
        L.append("| **mean** | " + " | ".join(f"{np.mean([T[(p, d, c)] for c in CASES if (p, d, c) in T]):.2f}" for d in ALL) + " |")
        L.append("")
        v2 = np.array([T[(p, "v2", c)] for c in CASES])
        for d in ALL:
            if d == "v2":
                continue
            x = np.array([T[(p, d, c)] for c in CASES])
            pv = stats.wilcoxon(v2, x, zero_method="zsplit").pvalue
            L.append(f"- v2 vs {d}: {v2.mean():.2f} vs {x.mean():.2f}; v2 better {int((v2 < x).sum())}/{len(CASES)}; p = {pv:.3f}")
        L.append("")
    L.append("MCPO run faithfully on the organisers' ReMIND2Reg files (from navigation; cases present in the release): "
             + ", ".join(f"{c[-3:]} ceT1 {float(mf[c]['mcpo_ceT1']):.2f} / T2 {mf[c]['mcpo_T2']}" for c in CASES if c in mf))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"run": do_run, "report": do_report}[sys.argv[1]]()
