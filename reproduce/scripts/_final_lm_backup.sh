#!/bin/bash
# 2026-11-05: before re-scoring with the final two-observer landmarks (C:\ReMIND_work\landmarks_final).
# Copies the paper state and the result tables; MOVES (never deletes) every per-landmark result of ReMIND and the
# external site computed with the provisional (one-observer) landmarks, so the scripts recompute them.
set -u
cd /mnt/c/ReMIND_work
S=fer/superseded_landmarks_2026-11-05
mkdir -p $S paper/build/provisional_landmarks_2026-11-05
cp -r fer/tables $S/tables
cp -r paper/MIA/latex paper/MIA/pdf paper/MIA/imagenes paper/build/provisional_landmarks_2026-11-05/
n=0
for d in f3d/f3d_mind f3d/f3d_lncc f3d/f3d_mind_lncc mcpo mcpo_t2 steps \
         extra_deformables/convexadam extra_deformables/convexadam_mask extra_deformables/f3d_crop extra_deformables/deeds extra_deformables/syn \
         extra_deformables_box/convexadam extra_deformables_box/convexadam_mask extra_deformables_box/f3d_crop extra_deformables_box/deeds extra_deformables_box/syn; do
  for coh in remind external; do
    if [ "$d" = steps ]; then src=fer/steps; pat="$coh/*.json"; else src=fer/$d; pat="$coh/*.json"; fi
    for f in $src/$pat; do
      [ -f "$f" ] || continue
      dst=$S/${f#fer/}; mkdir -p "$(dirname "$dst")"; mv "$f" "$dst"; n=$((n+1))
    done
  done
done
echo "moved $n json files to $S"
# AFFINE_VERIFY: drop the Subject-3 row (its affine is recomputed with the 3D T1)
/home/ncrhurh/anaconda3/envs/monai/bin/python - <<'PY2'
import csv
p = "fer/tables/AFFINE_VERIFY.csv"
rows = list(csv.DictReader(open(p)))
keep = [r for r in rows if not (r["cohort"] == "external" and r["case"] == "Subject-3")]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(keep)
print("AFFINE_VERIFY rows", len(rows), "->", len(keep))
PY2
