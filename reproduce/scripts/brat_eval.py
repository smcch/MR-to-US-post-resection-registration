"""External-site case (Subject-1, one pair MR pre + US post, landmarks by SANTI) through the SAME arms as the ReMIND
provisional comparison.  Independent of the manuscript; external images are for local use only.

Arms (configurations unchanged, landmark-blind until scoring):
  pose_search   frozen TIGHT/K12 FFT search (rg_search_tight.TIGHT, clamped descent), run here for the first time
  pose          + guarded 0.5 mm polish (annotation_eval.polish); gate: abstain (identity) if rotation >= 9.5 deg
  paper         shipped stage (rs_prestage: FEM basis in the posed frame, 30 modes + residual s1.5, lambda 3e-3;
                no tumour / iUS segmentation for this site -> no exclusion, as RESECT)
  v2            frozen v2 (v2_method.ACCEPTED on the accepted pose; FALLBACK from identity if abstained)
  nr            ReMIND2Reg NiftyReg baseline (reg_aladin rigid + affine, annotation_eval.niftyreg_arms)
  nr_f3d        that affine + reg_f3d MIND-SSC
  nr_paper / nr_v2   the NiftyReg affine + our paper / v2 stage (as v2_nr_affine_deform.py)
  mcpo          MCPO original implementation: the unmodified official script on inputs prepared as the ReMIND2Reg
                organisers prepare theirs (iUS and MR resampled into a 256^3 box at 0.5 mm, MR voxel axes, centred on
                the iUS), from navigation
Usage: python brat_eval.py run|score [Subject-k]      (default Subject-1; the site may contribute several cases)
Output: /mnt/c/ReMIND_work/brat/out/<case>/..., brat/out/<case>/REPORT.md and TRE_landmarks.csv
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rs_prestage as RSP                                            # noqa  (import before install: rebinding)
import rigid_search_fft as RS                                        # noqa
import rg_cohorts                                                    # noqa
import annotation_eval as E                                          # noqa
from annotation_qc import read_markups, horn                         # noqa

W = Path("/mnt/c/ReMIND_work")
CASE = sys.argv[2] if len(sys.argv) > 2 else "Subject-1"
D = W / "brat" / CASE
OUT = W / "brat/out"
SEG = D / f"{CASE}_mr_synthseg.nii.gz"
EMPTY = (np.zeros((0, 3)), np.zeros((0, 3)))
PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
MCPO = W / "scripts/third_party/MCPO_p4_6543.py"

rg_cohorts.COHORTS["brat_set"] = dict(
    cases=lambda: [CASE], paths=lambda c: (D / "us.nii.gz", D / "mr.nii.gz", None, [1]),
    landmarks=lambda c: (None, None), seg=lambda c: SEG, tumor=lambda c: None, basis=lambda c: None,
    frozen_arm=lambda c: None, lam_eig="3e-3", exclude_labels="1", exclude_dilate="3",
    regime="post", floor=None, label=f"external site, {CASE}")


def mcpo_original(out):
    """Unmodified official script; inputs prepared like the organisers' ReMIND2Reg files."""
    keep = out / "mcpo/disp.nii.gz"
    if keep.exists():
        return keep
    from reg_core import resample_to_grid
    usi, mri = nib.load(str(D / "us.nii.gz")), nib.load(str(D / "mr.nii.gz"))
    a = np.asarray(usi.dataobj)
    nz = np.argwhere(a > 0)
    w = nib.affines.apply_affine(usi.affine, np.r_[nz.min(0), nz.max(0)].reshape(2, 3))
    centre = w.mean(0)
    Rm = mri.affine[:3, :3] / np.linalg.norm(mri.affine[:3, :3], axis=0)
    aff = np.eye(4)
    aff[:3, :3] = Rm * 0.5
    aff[:3, 3] = centre - aff[:3, :3] @ np.full(3, 127.5)
    dims = (256, 256, 256)
    f = resample_to_grid(usi, aff, dims).astype(np.float32)
    m = resample_to_grid(mri, aff, dims).astype(np.float32)
    ind, outd = Path("/tmp/bratmcpo/in"), Path("/tmp/bratmcpo/out")   # no "_" in the path: the script splits it at the first "_"
    for p in (ind, outd):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
    nib.save(nib.Nifti1Image(f, aff), str(ind / "ReMIND2Reg_0900_0000.nii.gz"))
    nib.save(nib.Nifti1Image(m, aff), str(ind / "ReMIND2Reg_0900_0001.nii.gz"))
    (out / "mcpo").mkdir(parents=True, exist_ok=True)
    with open(out / "mcpo/mcpo.log", "w") as fh:
        rc = subprocess.run([PY, str(MCPO), "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    dp = outd / "disp_0900_0000_0900_0001.nii.gz"
    if not dp.exists():
        raise RuntimeError(f"MCPO produced no output (rc {rc})")
    shutil.copy2(dp, keep)
    (out / "mcpo/info.json").write_text(json.dumps(dict(rc=rc, box_affine=aff.tolist(), us_fraction_in_box=float((f > 0).sum() / max(1, (a > 0).sum())))))
    shutil.rmtree("/tmp/bratmcpo", ignore_errors=True)
    return keep


def do_run():
    import v2_remind_eval as VE
    import v2_method as FM
    import affine_cascade as AC
    from rg_search_tight import TIGHT, install_clamped_descend
    spec = rg_cohorts.install("brat_set")
    out = OUT / CASE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 1. frozen pose search (blind)
    sroot = OUT / "search"
    if not (sroot / "brat_set" / CASE / "aff.txt").exists():
        install_clamped_descend(TIGHT["rot_cap"])
        RS.search(CASE, "brat_set", dict(TIGHT), out_root=sroot)
    A0 = np.loadtxt(sroot / "brat_set" / CASE / "aff.txt")
    rot = E.rot_deg_of(A0)
    abstain = bool(rot >= E.SAT)
    np.savetxt(out / "aff_search.txt", A0)
    (out / "search_source.json").write_text(json.dumps(dict(rot_deg=rot, abstain=abstain, sat_deg=E.SAT)))
    print(f"[brat] search rot {rot:.2f} deg ({'ABSTAIN' if abstain else 'accepted'}), {time.time() - t0:.0f} s", flush=True)
    # 2. polish + shipped stage + identity fallback (landmark accessors blind)
    A1, pinfo = E.polish(CASE, A0, spec, out / "search_polish")
    print(f"[brat] polish move {pinfo['move_mm']:.2f} mm {'accepted' if pinfo['accepted'] else 'REJECTED'}", flush=True)
    RSP.case_landmarks = lambda d, c: EMPTY
    for tag, A in (("ours", A1), ("identity", np.eye(4))):
        r = RSP.prestage("brat_set", CASE, A, out / tag)
        if r.get("status") != "ok":
            raise RuntimeError(f"{tag}: {r}")
    wg = out / "identity/modal_residual_s15/US_fixed_grid.nii.gz"
    # 3. NiftyReg baseline (+ f3d MIND)
    E.niftyreg_arms(CASE, spec, wg, out)
    print(f"[brat] NiftyReg arms done", flush=True)
    # 4. frozen v2
    VE.paths = lambda case: (D / "us.nii.gz", D / "mr.nii.gz")
    VE.OUT = OUT
    if abstain:
        VE.deform(CASE, FM.FALLBACK, "v2", np.eye(4), out / "identity/basis_rig.npz")
    else:
        VE.deform(CASE, FM.ACCEPTED, "v2", A1, out / "ours/basis_rig.npz")
    # 5. NiftyReg affine + our stages (affine-safe relabel)
    A_nr = np.loadtxt(out / "niftyreg_l2r/aff.txt")
    RSP.relabel = AC.relabel
    VE.relabel = AC.relabel
    r = RSP.prestage("brat_set", CASE, A_nr, out / "nraff_paper")
    if r.get("status") != "ok":
        raise RuntimeError(f"nraff_paper: {r}")
    VE.deform(CASE, FM.ACCEPTED, "nraff_v2", A_nr, out / "nraff_paper/basis_rig.npz")
    # 6. MCPO original
    mcpo_original(out)
    (out / "DONE").write_text(time.strftime("%F %T"))
    print(f"[brat] RUN DONE ({time.time() - t0:.0f} s)", flush=True)


def disp_at(p, pts):
    return E.sample_disp(nib.load(str(p)), pts)


def do_score():
    out = OUT / CASE
    mr, _ = read_markups(D / f"{CASE}-MR.mrk.json")
    us, _ = read_markups(D / f"{CASE}-US-SANTI.mrk.json")
    ids = sorted(set(mr) & set(us))
    MR, US = np.array([mr[i] for i in ids]), np.array([us[i] for i in ids])
    src = json.loads((out / "search_source.json").read_text())
    A1 = np.loadtxt(out / "search_polish/aff.txt")
    A_nr = np.loadtxt(out / "niftyreg_l2r/aff.txt")
    ap = lambda A, p: p @ A[:3, :3].T + A[:3, 3]                              # noqa: E731
    Av2 = np.eye(4) if src["abstain"] else A1
    u_id = disp_at(out / "identity/modal_residual_s15/disp_mm.nii.gz", US)
    pred = dict(initial=US, nr=ap(A_nr, US), nr_f3d=US + disp_at(out / "niftyreg_l2r_f3d/disp_mm.nii.gz", US),
                pose=US if src["abstain"] else ap(A1, US),
                paper=(US + u_id) if src["abstain"] else ap(A1, US + disp_at(out / "ours/modal_residual_s15/disp_mm.nii.gz", US)),
                v2=ap(Av2, US + disp_at(out / "v2/disp_mm.nii.gz", US)),
                nr_paper=ap(A_nr, US + disp_at(out / "nraff_paper/modal_residual_s15/disp_mm.nii.gz", US)),
                nr_v2=ap(A_nr, US + disp_at(out / "nraff_v2/disp_mm.nii.gz", US)))
    ui = nib.load(str(out / "mcpo/disp.nii.gz"))
    u = np.asarray(ui.dataobj, np.float32)
    inv = np.linalg.inv(ui.affine)
    v = US @ inv[:3, :3].T + inv[:3, 3]
    from scipy.ndimage import map_coordinates
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    pred["mcpo"] = US + du @ ui.affine[:3, :3].T
    inside = int(np.all((v >= 0) & (v <= 255), 1).sum())
    err = {k: np.linalg.norm(p - MR, axis=1) for k, p in pred.items()}
    R, t = horn(US, MR)
    err["bestfit"] = np.linalg.norm(US @ R.T + t - MR, axis=1)
    lab = dict(initial="navigation (initial)", nr="NiftyReg baseline ReMIND2Reg (rigid+affine)", nr_f3d="NiftyReg + f3d MIND",
               mcpo="MCPO original", pose="our pose only", paper="paper method", v2="v2", nr_paper="NiftyReg affine + paper stage",
               nr_v2="NiftyReg affine + v2 stage", bestfit="best-fit rigid (landmark oracle)")
    meta = {k: json.loads((out / f"{k}/meta.json").read_text()) for k in ("v2", "nraff_v2")}
    L = [f"# External site, {CASE}: {len(ids)} landmarks (observer SANTI); pose rot {src['rot_deg']:.2f} deg, "
         f"{'ABSTAINED' if src['abstain'] else 'accepted'}", "", "| arm | mean | median | max |", "|---|---|---|---|"]
    for k in lab:
        L.append(f"| {lab[k]} | {err[k].mean():.2f} | {np.median(err[k]):.2f} | {err[k].max():.2f} |")
    L += ["", f"MCPO box: {inside}/{len(ids)} landmarks inside the 256^3 0.5 mm box.",
          f"v2 regularity: our pose fold {meta['v2']['fold_pct']:.3f} %, SDLogJ {meta['v2']['sdlogj']:.3f}; "
          f"NiftyReg affine fold {meta['nraff_v2']['fold_pct']:.3f} %, SDLogJ {meta['nraff_v2']['sdlogj']:.3f}"]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    with open(out / "TRE_landmarks.csv", "w") as f:
        f.write("lm," + ",".join(lab) + "\n")
        for j, i in enumerate(ids):
            f.write(f"{i}," + ",".join(f"{err[k][j]:.3f}" for k in lab) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": do_run, "score": do_score}[sys.argv[1]]()
