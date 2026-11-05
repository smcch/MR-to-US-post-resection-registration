#!/bin/bash
# Re-do an external-site case whose images/landmarks changed on G:: wipe every output of the case, stage again,
# SynthSeg, lean chain (manuscript arms).  Usage: bash brat_redo_case.sh Subject-3
set -u
C=$1
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
cd /mnt/c/ReMIND_work
mkdir -p logs/brat_$C/old_$(date +%Y%m%d_%H%M)
mv brat/$C brat/${C}_old_$(date +%Y%m%d_%H%M) 2>/dev/null
rm -rf brat/out/$C fer/run/external/$C fer/run_mcpopose/external/$C fer/mcpo_pose/external/$C fer/steps/external/$C.json \
       fer/mcpo/external/$C.json fer/f3d/f3d_mind/external/$C.json fer/f3d/f3d_lncc/external/$C.json fer/f3d/f3d_mind_lncc/external/$C.json \
       /home/ncrhurh/remind_biomech/brat_out/${C}_mr_synthseg.nii.gz
$PY scripts/brat_stage.py $C || exit 1
rm -rf /tmp/bratseg_$C && mkdir -p /tmp/bratseg_$C && cp brat/$C/mr.nii.gz /tmp/bratseg_$C/${C}_mr.nii.gz
(cd ~/remind_biomech && bash run_synthseg_gpu.sh /tmp/bratseg_$C /home/ncrhurh/remind_biomech/brat_out --robust > /mnt/c/ReMIND_work/logs/brat_${C}_synthseg.log 2>&1)
cp /home/ncrhurh/remind_biomech/brat_out/${C}_mr_synthseg.nii.gz brat/$C/${C}_mr_synthseg.nii.gz || exit 1
bash scripts/brat_run_case_lean.sh $C
echo "REDO DONE $C"
