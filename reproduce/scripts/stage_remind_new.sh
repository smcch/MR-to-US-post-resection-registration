#!/usr/bin/env bash
# Stage new ReMIND landmark cases onto C: (G: only READ; G: has bad blocks -> every copy is integrity-checked).
# Usage: bash stage_remind_new.sh ReMIND-036 ReMIND-038 ...
set -u
G=/mnt/g/ECOGRAFIA/ReMIND_biomech; A=/mnt/g/ECOGRAFIA/ReMIND_biomech_archive; W=/mnt/c/ReMIND_work
mkdir -p $W/archive/data/us $W/archive/data/synthseg_in/preop $W/archive/data/synthseg_out/preop \
         $W/archive/data/segmentations/MRI $W/archive/data/segmentations/US $W/annotation_remind/LANDMARKS \
         $W/annotation_remind/PILOT $W/remind/data $W/remind/segs $W/remind/landmarks
cpv() {  # copy + verify (cmp; gzip -t for .gz)
  [ -f "$1" ] || { echo "  absent on G: $1"; return; }
  cp -n "$1" "$2" 2>/dev/null
  d="$2"; [ -d "$2" ] && d="$2/$(basename "$1")"
  if cmp -s "$1" "$d"; then case "$d" in *.gz) gzip -t "$d" 2>/dev/null || echo "  CORRUPT gzip: $d";; esac
  else echo "  COPY/READ FAILED: $1"; fi
}
for c in "$@"; do
  echo "== $c"
  cpv $A/data/us/${c}_USpreimri.nii.gz $W/archive/data/us
  cpv $A/data/synthseg_in/preop/${c}_T2.nii.gz $W/archive/data/synthseg_in/preop
  cpv $A/data/synthseg_out/preop/${c}_T2_synthseg.nii.gz $W/archive/data/synthseg_out/preop
  cpv $A/data/segmentations/MRI/${c}-pre-mri-segmentation.nii.gz $W/archive/data/segmentations/MRI
  cpv $A/data/segmentations/US/${c}-post-us-segmentation.nii.gz $W/archive/data/segmentations/US
  for sub in search_tight/remind/remind search_tight_remind_full/base/remind; do
    if [ -f $G/results/rigid_gate/$sub/$c/aff.txt ]; then mkdir -p $W/results/rigid_gate/$sub/$c; cpv $G/results/rigid_gate/$sub/$c/aff.txt $W/results/rigid_gate/$sub/$c; echo "  pose from $sub"; fi
  done
  # the ReMIND helper tree used by contrib_factorial_remind / mcpo_pose_deform
  cp -n $W/archive/data/us/${c}_USpreimri.nii.gz $W/archive/data/synthseg_in/preop/${c}_T2.nii.gz $W/remind/data/ 2>/dev/null
  cp -n $W/archive/data/synthseg_out/preop/${c}_T2_synthseg.nii.gz $W/remind/segs/ 2>/dev/null
  cp -n $W/archive/data/segmentations/MRI/${c}-pre-mri-segmentation.nii.gz $W/archive/data/segmentations/US/${c}-post-us-segmentation.nii.gz $W/remind/segs/ 2>/dev/null
done
mkdir -p $W/results/rigid_gate/search_tight/remind $W/results/rigid_gate/search_tight_remind_full
cpv $G/results/rigid_gate/search_tight/remind/PER_CASE.csv $W/results/rigid_gate/search_tight/remind
cpv $G/results/rigid_gate/search_tight_remind_full/POSES.csv $W/results/rigid_gate/search_tight_remind_full
# ALL landmark files (also a safety copy: they live on the failing disk)
for f in $G/annotation_remind/LANDMARKS/*.mrk.json; do cpv "$f" $W/annotation_remind/LANDMARKS; cp -f "$W/annotation_remind/LANDMARKS/$(basename "$f")" $W/remind/landmarks/; done
for f in MR-023 US-olga-023 US-santi-023; do cpv $G/annotation_remind/PILOT/$f.mrk.json $W/annotation_remind/PILOT; done
ls $W/annotation_remind/LANDMARKS | wc -l; du -sh $W/archive
echo "STAGE NEW DONE"
