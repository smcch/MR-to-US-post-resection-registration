#!/bin/bash
# Writes the intra-observer report (UTF-8) to C:\ReMIND_work\annotation\intra_round2_<date>.md and prints it
cd /mnt/c/ReMIND_work/scripts
OUT=/mnt/c/ReMIND_work/annotation/intra_round2_$(date +%F).md
{
echo "# Intra-observer variability of US landmarks (observer SANTI), round 1 vs round 2, computed $(date +%F)"
echo "Intra = distance between rounds per landmark (mm); nav = shared MR vs US (navigation error). MR set fixed and shared; only US re-annotated. Script: scripts/_intra_summary.sh"
/home/ncrhurh/anaconda3/envs/monai/bin/python _intra_summary.py
} > $OUT
cat $OUT
