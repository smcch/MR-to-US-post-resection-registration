#!/usr/bin/env python
"""Target registration error (TRE) of an initialization and of its FE refinement at landmark pairs.

Landmarks: two 3D Slicer markups files (.mrk.json, as released in landmarks/ReMIND), one with the MR points and one
with the iUS points.  Points are paired by the integer that ends their label (MR-7 with US-7).  Coordinates in
the files are LPS (Slicer convention) and are converted to RAS, the convention of the NIfTI affines.

Errors reported per landmark pair, in mm:
  initial  |x_us - x_mr|                     the navigation alignment
  affine   |A x_us - x_mr|                   the initialization (--affine)
  refined  |A (x_us + u(x_us)) - x_mr|       the initialization followed by the FE refinement (--disp)
and, as a reference, the residual of the six-degree-of-freedom rigid transform fitted to the pairs (best-fit).

Usage: python score_landmarks.py --mr-landmarks ReMIND-024-MR.mrk.json --us-landmarks ReMIND-024-US.mrk.json
                                 --affine out/aff.txt --disp out/disp_mm.nii.gz [--csv errors.csv]
"""
import argparse
import csv
import json
import re

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def read_markups(path):
    """{landmark number: RAS point} from a Slicer markups file."""
    j = json.load(open(path, encoding="utf-8"))
    pts = {}
    for m in j["markups"]:
        s = np.array([-1.0, -1.0, 1.0]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            if c.get("positionStatus", "defined") != "defined":
                continue
            k = int(re.search(r"(\d+)\s*$", c["label"]).group(1))
            pts[k] = np.array(c["position"], float) * s
    return pts


def apply(A, p):
    return p @ A[:3, :3].T + A[:3, 3]


def disp_at(path, pts):
    """Trilinear sample of a displacement image (X, Y, Z, 3; world mm) at world points."""
    u = nib.load(str(path))
    a = np.asarray(u.dataobj, np.float32)
    inv = np.linalg.inv(u.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.stack([map_coordinates(a[..., k], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest") for k in range(3)], -1)


def horn(P, Q):
    """Least-squares rigid (R, t) with Q ~ P R^T + t."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mr-landmarks", required=True)
    ap.add_argument("--us-landmarks", required=True)
    ap.add_argument("--affine", required=True)
    ap.add_argument("--disp", help="disp_mm.nii.gz of fe_refine.py (omit to score the initialization only)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    mr, us = read_markups(a.mr_landmarks), read_markups(a.us_landmarks)
    ids = sorted(set(mr) & set(us))
    if not ids:
        raise SystemExit("no landmark pair shares a label number")
    M = np.array([mr[i] for i in ids]); U = np.array([us[i] for i in ids])
    A = np.loadtxt(a.affine)
    err = {"initial": np.linalg.norm(U - M, axis=1), "affine": np.linalg.norm(apply(A, U) - M, axis=1)}
    if a.disp:
        err["refined"] = np.linalg.norm(apply(A, U + disp_at(a.disp, U)) - M, axis=1)
    R, t = horn(U, M)
    err["best-fit rigid"] = np.linalg.norm(U @ R.T + t - M, axis=1)
    print(f"{len(ids)} landmark pairs")
    for k, e in err.items():
        print(f"  {k:15s} mean {e.mean():.2f} mm  median {np.median(e):.2f}  max {e.max():.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["landmark"] + list(err))
            for j, i in enumerate(ids):
                w.writerow([i] + [f"{err[k][j]:.3f}" for k in err])


if __name__ == "__main__":
    main()
