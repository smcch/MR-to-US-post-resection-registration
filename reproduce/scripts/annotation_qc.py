"""Annotation QC + inter-rater analysis for the ReMIND landmark study
(protocol v1.0, docs/ANNOTATION_PROTOCOL.md, sections 5-6).

Reads Slicer Markups (.mrk.json) for one stage directory (PILOT or MAIN):
    <case>_raterA_MR.mrk.json          frozen MR references (stage 1)
    <case>_raterA_US.mrk.json          iUS correspondences, rater A
    <case>_raterB_US.mrk.json          iUS correspondences, rater B
Correspondence is by the trailing integer of the point label (F-7,
o_US_044-7, ReMIND-023_US_Olga-7 -> 7), never by order.  Coordinates are
converted from the file's declared coordinateSystem (LPS in Slicer exports)
to RAS, the world convention of every NIfTI/affine in this project.

Pre-registered QC (protocol 5.1): a pair is excluded when the two raters'
iUS picks disagree by more than 3 x MAD of that case's disagreements (the
lm_select.py rule: median + 3 * 1.4826 * MAD, iterated) or by more than
5 mm absolute -- whichever is stricter.  Consensus = midpoint (5.2).  A case
with < 8 retained pairs is flagged under-annotated (5.3).

Reported (protocol 6.1, 6.3): inter-rater error per case and cohort,
initial (navigation) TRE on the consensus set, and the best-fit rigid
residual of the consensus set (landmark-only quantities; no method output
is read here).

Usage: python annotation_qc.py PILOT|MAIN
"""
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

W = Path("/mnt/c/ReMIND_work")
ARCH = Path("/mnt/g/ECOGRAFIA/ReMIND_biomech_archive")
K_MAD = 3.0
ABS_MM = 5.0
MIN_PAIRS = 8
LITERATURE_FLOOR = (1.89, 0.37)          # ReMIND2Reg quoted inter-rater figure


def horn(a, b):
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def read_markups(path):
    d = json.load(open(path))
    pts = {}
    meta = []
    for m in d["markups"]:
        cs = m.get("coordinateSystem", "LPS")
        for cp in m["controlPoints"]:
            lab = cp.get("label", "")
            mm = re.search(r"(\d+)\s*$", lab)
            if cp.get("positionStatus", "defined") != "defined" or cp.get("position") is None:
                meta.append((lab, "undefined"))
                continue
            p = np.asarray(cp["position"], float)
            if cs == "LPS":
                p = p * np.array([-1.0, -1.0, 1.0])
            elif cs != "RAS":
                raise ValueError(f"{path}: unknown coordinateSystem {cs}")
            if mm is None:
                meta.append((lab, "nolabelnumber"))
                continue
            k = int(mm.group(1))
            if k in pts:
                raise ValueError(f"{path}: duplicate landmark number {k}")
            pts[k] = p
    return pts, meta


def qc_pairs(d):
    """lm_select.py rule on the inter-rater disagreements, then the 5 mm cap."""
    keep = np.ones(len(d), bool)
    for _ in range(5):
        med = np.median(d[keep])
        mad = np.median(np.abs(d[keep] - med)) * 1.4826 + 1e-9
        new = d <= med + K_MAD * mad
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    thr_mad = float(np.median(d[keep]) + K_MAD * (np.median(np.abs(d[keep] - np.median(d[keep]))) * 1.4826 + 1e-9))
    keep = keep & (d <= ABS_MM)
    return keep, thr_mad


def tumour_distance_map(case):
    p = ARCH / f"data/segmentations/MRI/{case}-pre-mri-segmentation.nii.gz"
    if not p.exists():
        return None
    img = nib.load(str(p))
    m = np.asarray(img.dataobj) > 0
    if not m.any():
        return None
    dt = distance_transform_edt(~m, sampling=img.header.get_zooms()[:3])
    return img, dt


def dist_to_tumour(dm, pts):
    if dm is None:
        return np.full(len(pts), np.nan)
    img, dt = dm
    inv = np.linalg.inv(img.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    v = np.rint(v).astype(int)
    out = np.full(len(pts), np.nan)
    ok = np.all((v >= 0) & (v < np.array(img.shape[:3])), axis=1)
    out[ok] = dt[v[ok, 0], v[ok, 1], v[ok, 2]]
    return out


def main(stage):
    src = W / "annotation_remind" / stage
    out = W / "results/annotation_remind" / stage.lower()
    out.mkdir(parents=True, exist_ok=True)
    sel = [r["case"] for r in csv.DictReader(open(W / "results/annotation_remind/SELECTION.csv"))
           if r["decision"] == stage]
    pair_rows, case_rows, lines = [], [], []
    lines.append(f"# Annotation QC report -- {stage} (protocol v1.0)\n")
    lines.append(f"Cases expected: {len(sel)}: {', '.join(sel)}\n")
    for c in sel:
        fa_mr = src / f"{c}_raterA_MR.mrk.json"
        fa = src / f"{c}_raterA_US.mrk.json"
        fb = src / f"{c}_raterB_US.mrk.json"
        missing = [f.name for f in (fa_mr, fa, fb) if not f.exists()]
        if missing:
            lines.append(f"## {c}: MISSING {missing}\n")
            case_rows.append(dict(case=c, status="missing_files"))
            continue
        mr, m_mr = read_markups(fa_mr)
        ua, m_a = read_markups(fa)
        ub, m_b = read_markups(fb)
        ids = sorted(set(mr) & set(ua) & set(ub))
        not_in = {"MR": sorted((set(ua) | set(ub)) - set(mr)),
                  "A": sorted(set(mr) - set(ua)), "B": sorted(set(mr) - set(ub))}
        MR = np.array([mr[i] for i in ids])
        A = np.array([ua[i] for i in ids])
        B = np.array([ub[i] for i in ids])
        dAB = np.linalg.norm(A - B, axis=1)
        keep, thr_mad = qc_pairs(dAB)
        cons = 0.5 * (A + B)
        init = np.linalg.norm(cons - MR, axis=1)
        initA = np.linalg.norm(A - MR, axis=1)
        initB = np.linalg.norm(B - MR, axis=1)
        # numbering sanity: is identity the optimal assignment between A and B?
        D = np.linalg.norm(A[:, None] - B[None], axis=-1)
        ri, ci = linear_sum_assignment(D)
        swapped = [(ids[i], ids[j]) for i, j in zip(ri, ci) if i != j]
        # protocol rule: MR references >= 3 mm apart
        DM = np.linalg.norm(MR[:, None] - MR[None], axis=-1) + np.eye(len(MR)) * 1e9
        close = [(ids[i], ids[j], round(float(DM[i, j]), 2)) for i in range(len(ids))
                 for j in range(i + 1, len(ids)) if DM[i, j] < 3.0]
        dm = tumour_distance_map(c)
        dtum = dist_to_tumour(dm, MR)
        # best-fit rigid residual on the retained consensus pairs (landmark-only)
        if keep.sum() >= 4:
            R, t = horn(cons[keep], MR[keep])
            resid = np.linalg.norm((cons @ R.T + t) - MR, axis=1)
        else:
            resid = np.full(len(ids), np.nan)
        for n, i in enumerate(ids):
            pair_rows.append(dict(
                case=c, lm=i, keep=bool(keep[n]),
                d_AB_mm=round(float(dAB[n]), 3),
                dx_AB=round(float(A[n, 0] - B[n, 0]), 3), dy_AB=round(float(A[n, 1] - B[n, 1]), 3),
                dz_AB=round(float(A[n, 2] - B[n, 2]), 3),
                init_consensus_mm=round(float(init[n]), 3),
                init_A_mm=round(float(initA[n]), 3), init_B_mm=round(float(initB[n]), 3),
                bestfit_resid_mm=round(float(resid[n]), 3),
                dist_tumour_mm=round(float(dtum[n]), 2) if np.isfinite(dtum[n]) else "",
                mr_x=MR[n, 0], mr_y=MR[n, 1], mr_z=MR[n, 2],
                usA_x=A[n, 0], usA_y=A[n, 1], usA_z=A[n, 2],
                usB_x=B[n, 0], usB_y=B[n, 1], usB_z=B[n, 2],
                us_x=cons[n, 0], us_y=cons[n, 1], us_z=cons[n, 2]))
        nk = int(keep.sum())
        status = "ok" if nk >= MIN_PAIRS else "under_annotated"
        cr = dict(case=c, status=status, n_mr=len(mr), n_A=len(ua), n_B=len(ub),
                  n_matched=len(ids), n_kept=nk, n_excluded=len(ids) - nk,
                  thr_mad_mm=round(thr_mad, 2),
                  interrater_mean_all=round(float(dAB.mean()), 3),
                  interrater_mean=round(float(dAB[keep].mean()), 3),
                  interrater_median=round(float(np.median(dAB[keep])), 3),
                  interrater_max_all=round(float(dAB.max()), 3),
                  interrater_rms_x=round(float(np.sqrt(np.mean((A - B)[keep, 0] ** 2))), 3),
                  interrater_rms_y=round(float(np.sqrt(np.mean((A - B)[keep, 1] ** 2))), 3),
                  interrater_rms_z=round(float(np.sqrt(np.mean((A - B)[keep, 2] ** 2))), 3),
                  init_tre_mean=round(float(init[keep].mean()), 3),
                  init_tre_median=round(float(np.median(init[keep])), 3),
                  init_tre_max=round(float(init[keep].max()), 3),
                  init_tre_A=round(float(initA[keep].mean()), 3),
                  init_tre_B=round(float(initB[keep].mean()), 3),
                  bestfit_rigid_resid=round(float(np.nanmean(resid[keep])), 3),
                  n_within10mm_tumour=int(np.nansum(dtum < 10)) if np.isfinite(dtum).any() else "",
                  n_swapped_by_assignment=len(swapped), n_refs_closer_3mm=len(close),
                  undefined=len(m_mr) + len(m_a) + len(m_b))
        case_rows.append(cr)
        lines.append(f"## {c}  [{status}]\n")
        lines.append(f"- points: MR {len(mr)}, rater A {len(ua)}, rater B {len(ub)}, matched by id {len(ids)}"
                     + (f"; unmatched {not_in}" if any(not_in.values()) else "")
                     + (f"; undefined/unnumbered {m_mr + m_a + m_b}" if (m_mr or m_a or m_b) else ""))
        lines.append(f"- QC: threshold min(median+3·MAD = {thr_mad:.2f} mm, {ABS_MM:.0f} mm); excluded "
                     f"{len(ids) - nk}/{len(ids)}"
                     + (f" -> landmarks {[ids[n] for n in range(len(ids)) if not keep[n]]} with d_AB "
                        f"{[round(float(dAB[n]), 2) for n in range(len(ids)) if not keep[n]]} mm" if nk < len(ids) else ""))
        lines.append(f"- inter-rater (retained): mean {cr['interrater_mean']:.2f}, median {cr['interrater_median']:.2f} mm; "
                     f"all pairs mean {cr['interrater_mean_all']:.2f}, max {cr['interrater_max_all']:.2f} mm; "
                     f"RMS per axis x/y/z {cr['interrater_rms_x']:.2f}/{cr['interrater_rms_y']:.2f}/{cr['interrater_rms_z']:.2f} mm")
        lines.append(f"- initial TRE (navigation, consensus, retained): mean {cr['init_tre_mean']:.2f}, "
                     f"median {cr['init_tre_median']:.2f}, max {cr['init_tre_max']:.2f} mm "
                     f"(rater A alone {cr['init_tre_A']:.2f}, rater B alone {cr['init_tre_B']:.2f})")
        lines.append(f"- best-fit rigid residual of the consensus set: {cr['bestfit_rigid_resid']:.2f} mm")
        if np.isfinite(dtum).any():
            lines.append(f"- MR references within 10 mm of the preoperative tumour segmentation: "
                         f"{cr['n_within10mm_tumour']}/{len(ids)} (median distance {np.nanmedian(dtum):.1f} mm)")
        if swapped:
            lines.append(f"- NUMBERING CHECK: optimal A<->B assignment differs from identity for "
                         f"{len(swapped)} points {swapped[:6]} -- inspect labels")
        else:
            lines.append("- numbering check: identity is the optimal A<->B assignment (labels consistent)")
        if close:
            lines.append(f"- protocol rule >=3 mm between references violated by {len(close)} pairs: {close}")
        lines.append("")

    ok = [r for r in case_rows if r.get("status") in ("ok", "under_annotated")]
    if ok:
        pm = np.array([r["interrater_mean"] for r in ok])
        allk = [p for p in pair_rows if p["keep"]]
        dall = np.array([p["d_AB_mm"] for p in allk])
        dall_raw = np.array([p["d_AB_mm"] for p in pair_rows])
        ini = np.array([p["init_consensus_mm"] for p in allk])
        pim = np.array([r["init_tre_mean"] for r in ok])
        lines.append("## Cohort summary\n")
        lines.append(f"- cases {len(ok)} ({sum(r['status'] == 'ok' for r in ok)} valid, "
                     f"{sum(r['status'] == 'under_annotated' for r in ok)} under-annotated); "
                     f"pairs matched {len(pair_rows)}, retained {len(allk)} "
                     f"({len(pair_rows) - len(allk)} excluded by QC)")
        lines.append(f"- **inter-rater floor**: mean of per-case means {pm.mean():.2f} ± {pm.std(ddof=1) if len(pm) > 1 else 0:.2f} mm "
                     f"(range {pm.min():.2f}–{pm.max():.2f}); pooled retained pairs mean {dall.mean():.2f}, "
                     f"median {np.median(dall):.2f}, 90th pct {np.percentile(dall, 90):.2f} mm; "
                     f"pooled ALL pairs (before QC) mean {dall_raw.mean():.2f}, median {np.median(dall_raw):.2f} mm")
        lines.append(f"- literature reference (ReMIND2Reg, borrowed technique): {LITERATURE_FLOOR[0]:.2f} ± {LITERATURE_FLOOR[1]:.2f} mm")
        lines.append(f"- **initial TRE** (navigation, consensus): mean of per-case means {pim.mean():.2f} ± "
                     f"{pim.std(ddof=1) if len(pim) > 1 else 0:.2f} mm; pooled mean {ini.mean():.2f}, median {np.median(ini):.2f}, "
                     f"max {ini.max():.2f} mm")
        bf = np.array([r["bestfit_rigid_resid"] for r in ok])
        lines.append(f"- best-fit rigid residual (consensus, mean of cases): {bf.mean():.2f} mm")
        # split-half: how well does the consensus approximate each rater (equals half of d_AB by construction)
        lines.append(f"- consensus-to-rater distance = d_AB/2 by construction: mean {dall.mean() / 2:.2f} mm")
        lines.append("")

    with open(out / "PAIRS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    keys = []
    for r in case_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "CASES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(case_rows)
    # consensus evaluation set, one CSV per case (RAS mm), retained pairs only
    for c in sel:
        rows = [p for p in pair_rows if p["case"] == c and p["keep"]]
        if not rows:
            continue
        with open(out / f"consensus_{c}.csv", "w", newline="") as f:
            f.write("# consensus landmarks, RAS mm; us_* = midpoint of rater A/B iUS picks; "
                    "mr_* = frozen MR reference; protocol v1.0 QC applied\n")
            w = csv.DictWriter(f, fieldnames=["lm", "mr_x", "mr_y", "mr_z", "us_x", "us_y", "us_z"])
            w.writeheader()
            for p in rows:
                w.writerow({k: p[k] for k in w.fieldnames})
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PILOT")
