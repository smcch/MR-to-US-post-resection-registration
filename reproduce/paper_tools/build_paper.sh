#!/usr/bin/env bash
# Rebuild the FE-refinement manuscript from the result tables: numbers and tables -> statistics figure -> PDFs.
# The image figures (fig1, fig3, fig4) are rendered by their own scripts (see README.md); they do not change with the
# numbers unless the registrations themselves are re-run.
# Usage (WSL): bash /mnt/c/ReMIND_work/paper/tools/build_paper.sh
set -u
P=/mnt/c/ReMIND_work/paper
PY=/home/ncrhurh/anaconda3/envs/monai/bin/python
cd $P/tools
$PY build_references.py || exit 1
$PY build_numbers.py || exit 1
$PY fig_cases_stats.py || exit 1
cd $P/MIA/latex
for doc in main supplementary; do
  ~/tectonic --keep-logs --keep-intermediates $doc.tex > $P/build/tectonic_$doc.out 2>&1
  echo "tectonic $doc rc=$?"
  ~/tectonic --keep-logs --keep-intermediates $doc.tex >> $P/build/tectonic_$doc.out 2>&1   # second pass for references
  mv -f $doc.log $doc.aux $P/build/ 2>/dev/null
  rm -f $doc.out $doc.bbl $doc.blg $doc.spl $doc.xdv 2>/dev/null
  mv -f $doc.pdf $P/MIA/pdf/ 2>/dev/null
done
cd $P/tools
$PY check_build.py
