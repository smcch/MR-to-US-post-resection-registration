#!/bin/bash
# 2026-10-05: MCPO for ReMIND-009 (not in the ReMIND2Reg release) on our 0.5 mm box, ceT1 (default) and T2 variant;
# MCPO pose + FE refinement on it; rescoring; paper rebuild.
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
cd /mnt/c/ReMIND_work
$PY scripts/fer_mcpo.py --cohorts remind 2>&1 | grep -a "\[mcpo\]\|MCPO DONE\|Error\|error" | grep -v Warning
$PY scripts/fer_mcpo.py --cohorts remind --t2 2>&1 | grep -a "\[mcpo\]\|MCPO DONE\|Error\|error" | grep -v Warning
ls -la fer/mcpo_fields/remind/ReMIND-009/ 2>&1
$PY scripts/fer_mcpo_pose.py --cohorts remind --cases ReMIND-009 2>&1 | grep -a "\[mcpopose\]\|Error\|error" | grep -v Warning
cat fer/mcpo_pose/remind/ReMIND-009/fit.json; echo
$PY scripts/fer_score.py 2>&1 | grep -v WARNING | tail -n 6
$PY scripts/fer_steps.py report > /dev/null 2>&1; echo "steps report rc=$?"
