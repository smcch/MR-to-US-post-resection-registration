#!/bin/bash
# Add external-site cases end to end with the manuscript arms only: stage from G:, SynthSeg (GPU), register in the fer
# scripts, lean chain.  Waits first for a running lean chain of another case.  Usage: bash brat_add_case_lean.sh Subject-9 Subject-11 Subject-12
set -u
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
cd /mnt/c/ReMIND_work
while pgrep -f "brat_run_case_lean.sh" > /dev/null; do sleep 15; done
for C in "$@"; do
  $PY scripts/brat_stage.py $C || { echo "stage failed $C"; continue; }
  rm -rf /tmp/bratseg_$C && mkdir -p /tmp/bratseg_$C && cp brat/$C/mr.nii.gz /tmp/bratseg_$C/${C}_mr.nii.gz
  (cd ~/remind_biomech && bash run_synthseg_gpu.sh /tmp/bratseg_$C /home/ncrhurh/remind_biomech/brat_out --robust > /mnt/c/ReMIND_work/logs/brat_${C}_synthseg.log 2>&1)
  cp /home/ncrhurh/remind_biomech/brat_out/${C}_mr_synthseg.nii.gz brat/$C/${C}_mr_synthseg.nii.gz || { echo "synthseg failed $C"; continue; }
  $PY - <<EOF
import re
from pathlib import Path
for f in ("scripts/fer_common.py", "scripts/fer_run.py"):
    p = Path(f); s = p.read_text()
    m = re.search(r'^EXTERNAL = \[(.*)\]$', s, flags=re.M)
    cases = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
    if "$C" not in cases:
        cases.append("$C")
    s = s[:m.start()] + 'EXTERNAL = [' + ", ".join(f'"{c}"' for c in cases) + ']' + s[m.end():]
    p.write_text(s)
print("EXTERNAL:", cases)
EOF
  bash scripts/brat_run_case_lean.sh $C
done
echo "LEAN QUEUE DONE $*"
