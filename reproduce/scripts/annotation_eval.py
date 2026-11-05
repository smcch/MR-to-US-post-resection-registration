"""Method evaluation for the ReMIND landmark annotation study
(docs/ANNOTATION_PROTOCOL.md v1.0, section 6.4).  Two sub-commands:

  run   --stage PILOT|MAIN   compute every arm for the stage's cases.
                             LANDMARK-BLIND: no landmark file is read; the
                             withdrawn ReMIND landmark accessors are stubbed
                             to empty before any frozen module runs.
  score --stage PILOT|MAIN   TRE of every arm on the consensus set written
                             by annotation_qc.py.  Output goes to a SEALED
                             directory (protocol 3: raters must not see any
                             TRE while annotation is ongoing).

Arms (identical to the manuscript's comparator suite and to the shipped
method, no configuration changed):

  initial            navigation alignment, identity
  pose_search        frozen TIGHT/K12 search pose (results/rigid_gate/...)
  pose_polished      + guarded 0.5 mm polish (probe_halfmm rule: accept if
                     the RMS move over the FoV is <= 2 mm)
  deformation_only   frozen modal stage from the identity pose
                     (rs_prestage.prestage with A = I; the abstention
                     fallback of the manuscript)
  ours               polished pose -> relabel -> basis rebuilt in the posed
                     frame -> modal + residual (rs_prestage.prestage), gated:
                     abstain (= deformation_only) when the search rotation
                     is >= 9.5 deg, exactly as in the manuscript
  niftyreg_l2r       the Learn2Reg 2024 / ReMIND2Reg NiftyReg baseline exactly as the
                     manuscript runs it (l2r_baseline_cohorts.run_baseline: reg_aladin
                     -rmask FoV -noSym -nac -ln 2 -lp 10 on a 0.5 mm FoV box, rigid
                     then affine)
  niftyreg_l2r_f3d   reg_f3d --mindssc 1 -sx 5 -be 0.005 on the 1 mm working grid
                     with a FoV rmask, initialised with that affine
                     (l2r_f3d_cohorts.f3d_case)
  convexadam         convex_adam_pt defaults on the same working grid

TRE convention (validated in the manuscript): pred = A (x_us + u(x_us)) for
the modal arms, A x_us for the affine baseline, x_us + u for the f3d/ConvexAdam
fields on the working grid; error = |pred - x_mr|.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))

W = Path("/mnt/c/ReMIND_work")
R = W / "results"
ARCH = Path("/mnt/c/ReMIND_work/archive")
NR = Path("/home/ncrhurh/anaconda3/envs/niftyreg/bin")
SAT = 9.5
EMPTY = (np.zeros((0, 3)), np.zeros((0, 3)))
ARMS_DIR = R / "annotation_remind/arms"


# ------------------------------------------------------------------ helpers

def stage_cases(stage):
    sel = W / "results/annotation_remind/SELECTION.csv"
    return [r["case"] for r in csv.DictReader(open(sel)) if r["decision"] == stage]


def rot_deg_of(A):
    Rm = np.asarray(A)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def search_pose(case):
    """Frozen search pose: 14-case search first, then the full-collection
    search.  Returns (A, rot_deg from the search table, source path)."""
    for d, per in ((R / "rigid_gate/search_tight/remind/remind",
                    R / "rigid_gate/search_tight/remind/PER_CASE.csv"),
                   (R / "rigid_gate/search_tight_remind_full/base/remind",
                    R / "rigid_gate/search_tight_remind_full/POSES.csv")):
        f = d / case / "aff.txt"
        if f.exists():
            rot = None
            if per.exists():
                for r in csv.DictReader(open(per)):
                    if r["case"] == case and r.get("rot_deg"):
                        rot = float(r["rot_deg"])
            return np.loadtxt(f), rot, str(f)
    raise FileNotFoundError(f"no frozen search pose for {case}")


def run(cmd, log, retries=3):
    """NiftyReg occasionally segfaults (rc -11) and always succeeds on retry."""
    for k in range(retries):
        with open(log, "a") as fh:
            fh.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
            rc = subprocess.run([str(c) for c in cmd], stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc == 0:
            return
        print(f"    rc={rc} on try {k + 1}: {Path(str(cmd[0])).name}", flush=True)
    raise RuntimeError(f"{Path(str(cmd[0])).name} failed {retries}x")


def sample_disp(img, pts):
    inv = np.linalg.inv(img.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    d = np.asarray(img.dataobj, np.float32)
    return np.stack([map_coordinates(d[..., c], [v[:, 0], v[:, 1], v[:, 2]],
                                     order=1, mode="nearest") for c in range(3)], -1)


def apply_A(A, pts):
    return pts @ A[:3, :3].T + A[:3, 3]


# ------------------------------------------------------------------- run

def _install_remind_blind():
    """Bind the frozen modules to the ReMIND cohort with the landmark
    accessors stubbed to empty (the withdrawn set is never read)."""
    import rs_prestage                                              # noqa
    import rigid_search_fft                                         # noqa
    import rg_cohorts
    for k in ("remind", "remind113"):
        rg_cohorts.COHORTS[k] = dict(rg_cohorts.COHORTS[k],
                                     landmarks=lambda c: EMPTY)
    spec = rg_cohorts.install("remind")
    for m in ("rigid_search", "rigid_search_fft", "rs_prestage"):
        if m in sys.modules and hasattr(sys.modules[m], "case_landmarks"):
            sys.modules[m].case_landmarks = lambda d, c: EMPTY
    return spec


def polish(case, A0, spec, outdir):
    """Guarded 0.5 mm polish, verbatim rule of probe_halfmm.py (no landmark)."""
    import torch
    import rigid_search_fft as RS
    from capture_range import rotvec_deg, pose_to_affine
    from rg_search_tight import install_clamped_descend, TIGHT
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "aff.txt").exists() and (outdir / "polish.json").exists():
        return np.loadtxt(outdir / "aff.txt"), json.loads((outdir / "polish.json").read_text())
    install_clamped_descend(TIGHT["rot_cap"])
    fx, mv, ex, exl = spec["paths"](case)
    cc = RS.PoseCost(case, fx, mv, ex, exl, spacing=0.5, us_pts=None, mri_pts=None)
    c = np.asarray(cc.center, float)
    R0 = A0[:3, :3]
    r0 = rotvec_deg(R0)
    t0 = A0[:3, 3] + R0 @ c - c
    r1, t1, cost1 = cc.descend(r0, t0, levels=(1,), iters=(150,))
    A1 = pose_to_affine(r1, t1, c)
    ijk = np.argwhere(cc.fov)
    if len(ijk) > 4000:
        ijk = ijk[np.random.default_rng(0).choice(len(ijk), 4000, replace=False)]
    pts = (np.c_[ijk, np.ones(len(ijk))] @ cc.grid_aff.T)[:, :3]
    h = np.c_[pts, np.ones(len(pts))]
    move = float(np.sqrt(((h @ A1.T - h @ A0.T)[:, :3] ** 2).sum(1).mean()))
    keep = bool(move <= 2.0)
    A = A1 if keep else A0
    np.savetxt(outdir / "aff.txt", A)
    np.savetxt(outdir / "aff_halfmm_unguarded.txt", A1)
    info = dict(case=case, move_mm=move, accepted=keep, cost_halfmm=float(cost1),
                rot_search_deg=rot_deg_of(A0), rot_polished_deg=rot_deg_of(A))
    (outdir / "polish.json").write_text(json.dumps(info, indent=2))
    del cc
    torch.cuda.empty_cache()
    return A, info


def build_rmask(wg_path, seg_path, labels, out_path, dilate_mm=3.0):
    """FoV of the working grid minus the iUS segmentation labels dilated by
    dilate_mm (physical mm) -- mirrors reg_modal's own --exclude rule."""
    from reg_core import resample_to_grid
    g = nib.load(str(wg_path))
    fov = np.asarray(g.dataobj) > 0
    ex = np.zeros(fov.shape, bool)
    if seg_path is not None and Path(seg_path).exists():
        img = nib.load(str(seg_path))
        lab = np.isin(np.asarray(img.dataobj), labels).astype(np.float32)
        ex = resample_to_grid(img, g.affine, fov.shape, mode="nearest", data=lab) > 0.5
        if ex.any() and dilate_mm > 0:
            zooms = np.linalg.norm(g.affine[:3, :3], axis=0)
            ex = distance_transform_edt(~ex, sampling=tuple(float(z) for z in zooms)) <= dilate_mm
    mask = (fov & ~ex).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, g.affine), str(out_path))
    return int(mask.sum()), int((fov & ex).sum())


def niftyreg_arms(case, spec, wg_path, outdir):
    """The manuscript's NiftyReg comparators: the Learn2Reg challenge baseline
    and that baseline + f3d MIND-SSC (identical recipes to the cohort runs)."""
    from l2r_baseline_cohorts import run_baseline
    from l2r_f3d_cohorts import f3d_case
    fx, mv, ex, exl = spec["paths"](case)
    run_baseline(fx, mv, outdir / "niftyreg_l2r", crop_fov=True)
    f3d_case(fx, mv, wg_path, outdir / "niftyreg_l2r/aff.txt", outdir / "niftyreg_l2r_f3d")


def convexadam_arm(case, spec, wg_path, outdir):
    from convexAdam.convex_adam_MIND import convex_adam_pt
    from reg_core import resample_to_grid
    cd = outdir / "convexadam"
    cd.mkdir(parents=True, exist_ok=True)
    if (cd / "disp_vox.nii.gz").exists():
        return
    fx, mv, ex, exl = spec["paths"](case)
    wg = nib.load(str(wg_path))
    mvr = resample_to_grid(nib.load(str(mv)), wg.affine, wg.shape[:3]).astype(np.float32)
    t0 = time.time()
    u = np.asarray(convex_adam_pt(np.asarray(wg.dataobj, np.float32), mvr), np.float32)
    nib.save(nib.Nifti1Image(u, wg.affine), str(cd / "disp_vox.nii.gz"))
    (cd / "info.json").write_text(json.dumps(dict(wall_s=time.time() - t0,
                                                  note="voxel displacement on the 1 mm axis-aligned "
                                                       "working grid; pred = x_us + u(x_us)")))


def do_run(stage, force=False, only=None):
    spec = _install_remind_blind()
    import rs_prestage as RSP
    cases = only or stage_cases(stage)
    print(f"[eval] {stage}: {len(cases)} cases -> {ARMS_DIR}", flush=True)
    for case in cases:
        t0 = time.time()
        out = ARMS_DIR / case
        out.mkdir(parents=True, exist_ok=True)
        try:
            A0, rot_tab, src = search_pose(case)
            rot = rot_deg_of(A0)
            np.savetxt(out / "aff_search.txt", A0)
            (out / "search_source.json").write_text(json.dumps(
                dict(source=src, rot_deg=rot, rot_deg_table=rot_tab,
                     abstain=bool(rot >= SAT), sat_deg=SAT)))
            print(f"[eval] {case}: search rot {rot:.2f} deg "
                  f"({'ABSTAIN' if rot >= SAT else 'accepted'}) from {src}", flush=True)
            # 1. guarded polish (only meaningful for accepted poses; computed anyway)
            A1, pinfo = polish(case, A0, spec, out / "search_polish")
            print(f"[eval] {case}: polish move {pinfo['move_mm']:.2f} mm "
                  f"{'accepted' if pinfo['accepted'] else 'REJECTED'}", flush=True)
            # 2. shipped composition from the polished pose
            r = RSP.prestage("remind", case, A1, out / "ours", force=force)
            if r.get("status") != "ok":
                raise RuntimeError(f"ours: {r}")
            # 3. deformation-only (identity pose): the abstention fallback
            r = RSP.prestage("remind", case, np.eye(4), out / "identity", force=force)
            if r.get("status") != "ok":
                raise RuntimeError(f"identity: {r}")
            wg = out / "identity/modal_residual_s15/US_fixed_grid.nii.gz"
            # 4. comparators on the same 1 mm working grid
            niftyreg_arms(case, spec, wg, out)
            convexadam_arm(case, spec, wg, out)
            (out / "DONE").write_text(time.strftime("%F %T"))
            print(f"[eval] {case}: all arms done ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:                                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[eval] {case} FAILED: {e}", flush=True)
    print("RUN DONE", flush=True)


# ------------------------------------------------------------------- score

def load_pairs(stage):
    p = R / "annotation_remind" / stage.lower() / "PAIRS.csv"
    rows = list(csv.DictReader(open(p)))
    out = {}
    for r in rows:
        if r["keep"] != "True":
            continue
        c = r["case"]
        d = out.setdefault(c, dict(lm=[], mr=[], us=[], usA=[], usB=[]))
        d["lm"].append(int(r["lm"]))
        d["mr"].append([float(r[k]) for k in ("mr_x", "mr_y", "mr_z")])
        d["us"].append([float(r[k]) for k in ("us_x", "us_y", "us_z")])
        d["usA"].append([float(r[k]) for k in ("usA_x", "usA_y", "usA_z")])
        d["usB"].append([float(r[k]) for k in ("usB_x", "usB_y", "usB_z")])
    for c in out:
        for k in ("mr", "us", "usA", "usB"):
            out[c][k] = np.asarray(out[c][k], float)
    return out


def arm_predictions(case, us):
    """dict arm -> predicted MR-world positions of the iUS points `us`."""
    out = ARMS_DIR / case
    src = json.loads((out / "search_source.json").read_text())
    A0 = np.loadtxt(out / "aff_search.txt")
    A1 = np.loadtxt(out / "search_polish/aff.txt")
    u_ours = nib.load(str(out / "ours/modal_residual_s15/disp_mm.nii.gz"))
    u_id = nib.load(str(out / "identity/modal_residual_s15/disp_mm.nii.gz"))
    preds = {"initial": us.copy(),
             "pose_search": apply_A(A0, us),
             "pose_polished": apply_A(A1, us),
             "deformation_only": us + sample_disp(u_id, us),
             "ours_ungated": apply_A(A1, us + sample_disp(u_ours, us))}
    preds["ours"] = preds["deformation_only"] if src["abstain"] else preds["ours_ungated"]
    A_nr = np.loadtxt(out / "niftyreg_l2r/aff.txt")
    preds["niftyreg_l2r"] = apply_A(A_nr, us)
    preds["niftyreg_l2r_f3d"] = us + sample_disp(nib.load(str(out / "niftyreg_l2r_f3d/disp_mm.nii.gz")), us)
    ca = nib.load(str(out / "convexadam/disp_vox.nii.gz"))
    preds["convexadam"] = us + sample_disp(ca, us)      # 1 mm axis-aligned grid: vox = mm
    return preds, src


ARM_ORDER = ["initial", "niftyreg_l2r", "niftyreg_l2r_f3d", "convexadam",
             "deformation_only", "pose_search", "pose_polished", "ours_ungated", "ours"]


def do_score(stage):
    pairs = load_pairs(stage)
    sealed = R / "annotation_remind" / stage.lower() / "SEALED"
    sealed.mkdir(parents=True, exist_ok=True)
    prow, crow = [], []
    for case in sorted(pairs):
        if not (ARMS_DIR / case / "DONE").exists():
            print(f"[score] {case}: arms not finished, skipped", flush=True)
            continue
        d = pairs[case]
        res = {}
        for tag, us in (("consensus", d["us"]), ("raterA", d["usA"]), ("raterB", d["usB"])):
            preds, src = arm_predictions(case, us)
            res[tag] = {a: np.linalg.norm(preds[a] - d["mr"], axis=1) for a in ARM_ORDER}
        init = res["consensus"]["initial"]
        k = max(1, int(round(0.3 * len(init))))
        hard = np.argsort(-init)[:k]
        pinfo = json.loads((ARMS_DIR / case / "search_polish/polish.json").read_text())
        cr = dict(case=case, n=len(init), abstain=src["abstain"], rot_search_deg=round(src["rot_deg"], 2),
                  polish_accepted=pinfo["accepted"], polish_move_mm=round(pinfo["move_mm"], 2))
        for a in ARM_ORDER:
            e = res["consensus"][a]
            cr[f"{a}_mean"] = round(float(e.mean()), 3)
            cr[f"{a}_median"] = round(float(np.median(e)), 3)
            cr[f"{a}_max"] = round(float(e.max()), 3)
            cr[f"{a}_tre30"] = round(float(e[hard].mean()), 3)
            cr[f"{a}_raterA"] = round(float(res["raterA"][a].mean()), 3)
            cr[f"{a}_raterB"] = round(float(res["raterB"][a].mean()), 3)
        crow.append(cr)
        for n, lm in enumerate(d["lm"]):
            pr = dict(case=case, lm=lm, hard30=bool(n in set(hard.tolist())))
            for a in ARM_ORDER:
                pr[a] = round(float(res["consensus"][a][n]), 3)
            prow.append(pr)
    if not crow:
        print("[score] nothing to score", flush=True)
        return
    with open(sealed / "TRE_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)
    with open(sealed / "TRE_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    lines = [f"# SEALED method evaluation -- {stage} (protocol v1.0 section 6.4)\n",
             "Do not open while annotation is ongoing (protocol section 3: raters never see any TRE).\n",
             f"cases {len(crow)}, retained pairs {len(prow)}; abstained {sum(c['abstain'] for c in crow)}; "
             f"polish rejected {sum(not c['polish_accepted'] for c in crow)}\n",
             "| arm | mean of case means | pooled mean | pooled median | TRE30 (case-avg) | rater A | rater B |",
             "|---|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        cm = np.array([c[f"{a}_mean"] for c in crow])
        pooled = np.array([p[a] for p in prow])
        t30 = np.array([c[f"{a}_tre30"] for c in crow])
        ra = np.array([c[f"{a}_raterA"] for c in crow])
        rb = np.array([c[f"{a}_raterB"] for c in crow])
        lines.append(f"| {a} | {cm.mean():.2f} ± {cm.std(ddof=1) if len(cm) > 1 else 0:.2f} | {pooled.mean():.2f} | "
                     f"{np.median(pooled):.2f} | {t30.mean():.2f} | {ra.mean():.2f} | {rb.mean():.2f} |")
    lines.append("\nPer case (consensus, mean mm):\n")
    lines.append("| case | n | rot | abst | polish | " + " | ".join(ARM_ORDER) + " |")
    lines.append("|---|---|---|---|---|" + "---|" * len(ARM_ORDER))
    for c in crow:
        lines.append(f"| {c['case']} | {c['n']} | {c['rot_search_deg']} | {c['abstain']} | "
                     f"{'ok' if c['polish_accepted'] else 'REJ'} {c['polish_move_mm']} | "
                     + " | ".join(f"{c[a + '_mean']:.2f}" for a in ARM_ORDER) + " |")
    # paired Wilcoxon (per-case means), ours vs each comparator -- exploratory at pilot n
    try:
        from scipy.stats import wilcoxon
        lines.append("\nPaired Wilcoxon on per-case means (ours vs arm), zsplit; exploratory at this n:\n")
        for a in ("initial", "niftyreg_l2r", "niftyreg_l2r_f3d", "convexadam", "deformation_only", "pose_polished"):
            x = np.array([c["ours_mean"] for c in crow])
            y = np.array([c[f"{a}_mean"] for c in crow])
            if len(x) < 2 or np.allclose(x, y):
                lines.append(f"- vs {a}: n/a")
                continue
            p = wilcoxon(x, y, zero_method="zsplit").pvalue
            lines.append(f"- vs {a}: ours {x.mean():.2f} vs {y.mean():.2f} mm, p = {p:.3f}, "
                         f"ours better in {(x < y).sum()}/{len(x)} cases")
    except Exception as e:                                            # noqa: BLE001
        lines.append(f"(wilcoxon unavailable: {e})")
    (sealed / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (sealed / "README_SEALED.txt").write_text(
        "Sealed per protocol v1.0 section 3 (blinding): method TREs must not reach the raters\n"
        "while annotation (pilot re-annotation or MAIN) is ongoing. Opening this directory is a\n"
        "protocol deviation to be recorded in the CHANGELOG of docs/ANNOTATION_PROTOCOL.md.\n")
    print(f"[score] wrote {sealed}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--stage", default="PILOT")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a.stage, force=a.force, only=a.case)
    else:
        do_score(a.stage)
