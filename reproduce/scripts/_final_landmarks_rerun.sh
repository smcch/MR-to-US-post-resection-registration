#!/bin/bash
# 2026-11-05: re-run / re-score everything that depends on the ReMIND and external landmarks with the final
# two-observer set (fer_common.LANDMARK_SET = "final"), and redo external Subject-3 on its 3D T1.
# Prerequisite: scripts/_final_lm_backup.sh (moves the provisional per-landmark results away).
set -u
cd /mnt/c/ReMIND_work
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
L=logs/final_lm_2026-11-05; mkdir -p $L
echo "START $(date)"
# 1. Subject-3 from scratch on the 3D T1 (stage, SynthSeg, affine, MCPO, refinement, f3d/MCPO/steps of the external site)
bash scripts/brat_redo_case.sh Subject-3 > $L/1_redo_subject3.log 2>&1; echo "1 redo Subject-3 done $(date)"; tail -n 12 $L/1_redo_subject3.log
# 2. f3d (three similarity terms) on ReMIND
$PY scripts/fer_f3d.py --kinds f3d_mind f3d_lncc f3d_mind_lncc --cohorts remind > $L/2_f3d_remind.log 2>&1; echo "2 f3d remind rc=$? $(date)"
# 3. MCPO on ReMIND: re-scored from the stored fields (ReMIND-009 re-run on its own box), default ceT1 and T2 variant
$PY scripts/fer_mcpo.py --cohorts remind > $L/3_mcpo_remind.log 2>&1; echo "3 mcpo remind rc=$? $(date)"
$PY scripts/fer_mcpo.py --t2 --cohorts remind > $L/3b_mcpo_t2_remind.log 2>&1; echo "3b mcpo t2 remind rc=$? $(date)"
# 4. the other deformables (fields are not kept, so they are re-run), both field-of-view treatments
for CROP in cone box; do
  for M in convexadam convexadam_mask f3d_crop deeds syn; do
    $PY scripts/fer_extra_deformables.py --method $M --crop $CROP --cohorts remind external > $L/4_${M}_${CROP}.log 2>&1
    echo "4 $M $CROP: $(grep -a -c 'FAILED' $L/4_${M}_${CROP}.log) failed $(date)"
  done
done
# 5. affine first-run record (ReMIND-006/036) at the final landmarks
$PY scripts/fer_fix_affines.py > $L/5_fix_affines.log 2>&1; echo "5 fix_affines rc=$?"
# 6. scoring, steps, summaries
$PY scripts/fer_score.py > $L/6_score.log 2>&1; echo "6 score rc=$?"; tail -n 8 $L/6_score.log
$PY scripts/fer_steps.py run > $L/7_steps_run.log 2>&1; echo "7 steps run rc=$? $(date)"
$PY scripts/fer_steps.py report > $L/8_steps_report.log 2>&1; echo "8 steps report rc=$?"
$PY scripts/fer_extra_summary.py cone > $L/9_extra_cone.log 2>&1; $PY scripts/fer_extra_summary.py box > $L/9_extra_box.log 2>&1
$PY scripts/brat_summary.py > $L/10_brat_summary.log 2>&1
echo "ALL DONE $(date)"
