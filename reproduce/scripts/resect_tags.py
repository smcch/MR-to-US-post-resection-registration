"""RESECT .tag landmark adapter (MINC Tag Point File -> world-mm point pairs).

COORDINATE CONVENTION (determined empirically 2026-08-09, see below):
  * Each data line holds 6 floats: x1 y1 z1  x2 y2 z2  ["label"].
  * Column 1 (Volume1) = the FIRST volume named in the filename; for
    CaseN-MRI-afterUS.tag that is the **MRI** point. Column 2 (Volume2) is the
    **US-after** point.
  * The coordinates are MINC world mm and map to the NIfTI conversions' world
    frame with the IDENTITY transform — no RAS/LPS negation of x,y is needed.
    (MINC world is RAS+, same as the NIfTI sform of these conversions.)

Empirical validation (all 20 MRI-afterUS cases, 281 landmark pairs):
  * identity: 100% of both columns fall inside both volume bounds;
    negating x,y throws 53-100% of the points outside the US volume. Decisive.
  * Column assignment: sampling the US-after intensity at the points, column 2
    lies on nonzero US signal for 281/281 landmarks, column 1 only for 90%
    (worst case Case12, 19.3 mm shift: 47%) — US landmarks are picked on
    visible US signal, so column 2 is the US point.
  * Initial TRE (||MRI - US||) per case: 1.7-19.3 mm, mean of case means
    5.9 mm — consistent with published RESECT MRI<->after-US misalignments.

Usage as a library:
    from resect_tags import load_tag_pairs
    mri_pts, us_pts = load_tag_pairs("Case1-MRI-afterUS.tag")

CLI: python resect_tags.py file.tag  -> prints n and initial TRE stats.
"""
import re
import sys

import numpy as np


def parse_tag(path):
    """Parse an MNI Tag Point File with Volumes = 2.

    Returns (points1, points2): float64 arrays (n, 3) in world mm,
    in file order (column 1 = Volume1, column 2 = Volume2)."""
    txt = open(path).read()
    if "Points =" not in txt:
        raise ValueError(f"{path}: no 'Points =' section (not a tag file?)")
    body = txt.split("Points =", 1)[1]
    rows = []
    for line in body.splitlines():
        # strip trailing quoted label if any, keep numeric fields
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 6:
            rows.append([float(v) for v in nums[:6]])
    if not rows:
        raise ValueError(f"{path}: no point rows parsed")
    a = np.asarray(rows, dtype=np.float64)
    return a[:, :3], a[:, 3:6]


def load_tag_pairs(path):
    """For a CaseN-MRI-afterUS.tag file: returns (mri_pts, us_pts) in world mm
    of the NIfTI conversions (identity mapping, see module docstring)."""
    p1, p2 = parse_tag(path)
    return p1, p2  # Volume1 = MRI, Volume2 = US (filename order)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        mri, us = load_tag_pairs(f)
        d = np.linalg.norm(mri - us, axis=1)
        print(f"{f}: n={len(d)} initial TRE mean {d.mean():.2f} "
              f"sd {d.std():.2f} min {d.min():.2f} max {d.max():.2f} mm")
