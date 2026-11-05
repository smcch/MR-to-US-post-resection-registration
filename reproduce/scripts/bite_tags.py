"""BITE .tag adapter (MNI Tag Point File -> world-mm point pairs).

Same file format as RESECT (scripts/resect_tags.py) but the column convention is
RE-VALIDATED for BITE independently (see results/bite/BITE_PREP.md):

  * group4 `<pid>/3D/Tags.tag`: columns 1-3 = MR point, columns 4-6 = US point.
    Authority: the dataset's own MATLAB loader `group4/load_US_MR_landmarks.m`
        [mx,my,mz,ux,uy,uz,~] = textread(tag,'%f %f %f %f %f %f %s','headerlines',6)
    plus empirical checks (US-volume support / intensity).
  * group2 `<pid>_set{1,2,3}.tag` and `<pid>_all.tag`: columns 1-3 = MR
    (Talairach), columns 4-6 = US (Talairach).

  * MINC world mm map to the NIfTI world frame of our conversions with the
    IDENTITY transform (MINC world is RAS+, same as the NIfTI sform).
"""
import re
import sys

import numpy as np

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_tag(path):
    """Parse an MNI Tag Point File with Volumes = 2 -> (pts1, pts2) (n,3)."""
    txt = open(path).read()
    if "Points =" not in txt:
        raise ValueError(f"{path}: no 'Points =' section")
    body = txt.split("Points =", 1)[1]
    rows = []
    for line in body.splitlines():
        nums = NUM.findall(line)
        if len(nums) >= 6:
            rows.append([float(v) for v in nums[:6]])
    if not rows:
        raise ValueError(f"{path}: no point rows parsed")
    a = np.asarray(rows, dtype=np.float64)
    return a[:, :3], a[:, 3:6]


def load_tag_pairs(path):
    """(mr_pts, us_pts) in world mm."""
    return parse_tag(path)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        mr, us = load_tag_pairs(f)
        d = np.linalg.norm(mr - us, axis=1)
        print(f"{f}: n={len(d)} mTRE {d.mean():.2f} sd {d.std(ddof=1):.2f} "
              f"min {d.min():.2f} max {d.max():.2f} mm")
