#!/bin/bash
# Figures 4, 5 and 6 with the author's slices (2026-10-05): cavity visible, landmarks of the proposed method clearly better.
cd /mnt/c/ReMIND_work/paper/tools
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
$PY fig_qualitative.py resect_after:Case15:55 bite_g4:03:52 remind:ReMIND-024:52 remind:ReMIND-021:79 2>&1 | grep -v Warning | tail -5
$PY fig_fields.py resect_after:Case15:55 bite_g4:03:52 remind:ReMIND-024:52 2>&1 | grep -v Warning | tail -4
$PY fig_qualitative.py external:Subject-4:50 --stem fig6_external 2>&1 | grep -v Warning | tail -3
