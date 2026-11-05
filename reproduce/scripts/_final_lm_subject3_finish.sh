#!/bin/bash
# 2026-11-05: finish external Subject-3 on its 3D T1 after the master re-run (its affine had crashed with 8 threads;
# l2r_baseline_cohorts.run_baseline now retries with 2 threads, same matrix), then the final scoring.
set -u
cd /mnt/c/ReMIND_work
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
L=logs/final_lm_2026-11-05; C=Subject-3
echo "START $(date)"
$PY scripts/brat_lean.py $C > logs/brat_$C/1_lean_retry.log 2>&1; echo "lean rc=$?"; tail -n 3 logs/brat_$C/1_lean_retry.log
$PY scripts/fer_run.py --cohorts external --cases $C > $L/s3_fer_run.log 2>&1; echo "fer_run rc=$?"
$PY scripts/fer_f3d.py --kinds f3d_mind f3d_lncc f3d_mind_lncc --cohorts external > $L/s3_f3d.log 2>&1; echo "f3d rc=$?"
$PY scripts/fer_mcpo.py --cohorts external > $L/s3_mcpo.log 2>&1; echo "mcpo rc=$?"
$PY scripts/fer_mcpo_pose.py --cohorts external --cases $C > $L/s3_mcpo_pose.log 2>&1; echo "mcpo_pose rc=$?"
for CROP in cone box; do
  for M in convexadam convexadam_mask f3d_crop deeds syn; do
    $PY scripts/fer_extra_deformables.py --method $M --crop $CROP --cohorts external --cases $C > $L/s3_${M}_${CROP}.log 2>&1
    echo "extra $M $CROP: $(grep -a -c FAILED $L/s3_${M}_${CROP}.log) failed"
  done
done
$PY scripts/fer_score.py > $L/f_score.log 2>&1; echo "score rc=$?"; tail -n 8 $L/f_score.log
$PY scripts/fer_steps.py run > $L/f_steps_run.log 2>&1; echo "steps run rc=$?"
$PY scripts/fer_steps.py report > $L/f_steps_report.log 2>&1; echo "steps report rc=$?"
$PY scripts/fer_extra_summary.py cone > $L/f_extra_cone.log 2>&1; $PY scripts/fer_extra_summary.py box > $L/f_extra_box.log 2>&1
$PY scripts/brat_summary.py > $L/f_brat_summary.log 2>&1
echo "FINISH DONE $(date)"
