#!/bin/bash
# External-site case, manuscript arms only.  Needs the case staged + segmented + listed in fer_common/fer_run EXTERNAL.
# Usage: bash brat_run_case_lean.sh Subject-7
set -u
C=$1
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
cd /mnt/c/ReMIND_work
L=logs/brat_$C; mkdir -p $L
$PY scripts/brat_lean.py $C > $L/1_lean.log 2>&1; echo "lean rc=$?"
$PY scripts/fer_run.py --cohorts external --cases $C > $L/3_fer_run.log 2>&1; echo "fer_run rc=$?"
$PY scripts/fer_f3d.py --kinds f3d_mind f3d_lncc f3d_mind_lncc --cohorts external > $L/4_f3d.log 2>&1; echo "f3d rc=$?"
$PY scripts/fer_mcpo.py --cohorts external > $L/5_mcpo.log 2>&1; echo "mcpo rc=$?"
$PY scripts/fer_mcpo_pose.py --cohorts external --cases $C > $L/6_mcpo_pose.log 2>&1; echo "mcpo_pose rc=$?"
$PY scripts/fer_score.py > $L/7_score.log 2>&1; echo "score rc=$?"
$PY scripts/fer_steps.py run > $L/8_steps.log 2>&1; echo "steps rc=$?"
$PY scripts/fer_steps.py report > $L/9_steps_report.log 2>&1; echo "steps report rc=$?"
$PY scripts/brat_summary.py > logs/brat_summary.log 2>&1
echo "BRAT CASE DONE $C"
