#!/bin/bash
# Additional experiment (2026-10-05): four deformables on the challenge affine, MR cut to the iUS box, post-resection.
# Usage: bash _extra_run_all.sh [cone|box]
cd /mnt/c/ReMIND_work
CROP=${1:-cone}
MONAI=/home/ncrhurh/anaconda3/envs/monai/bin/python
D=fer/extra_deformables; [ "$CROP" = box ] && D=fer/extra_deformables_box
mkdir -p $D/logs
for m in convexadam f3d_crop deeds syn; do
  $MONAI scripts/fer_extra_deformables.py --method $m --crop $CROP > $D/logs/$m.log 2>&1
  grep -a "DONE\|FAILED" $D/logs/$m.log | tail -n 3
done
echo "ALL EXTRA DONE ($CROP)"
