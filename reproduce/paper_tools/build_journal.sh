#!/bin/bash
# Additional PDF in the journal layout of Elsevier's elsarticle class (final, 5p, two columns, Times), generated from
# main.tex: same content, floats widened to both columns, no line numbers.  Output: MIA/pdf/main_journal.pdf
# (the review version main.pdf is untouched).  The generated main_journal.tex is kept next to main.tex.
set -e
cd /mnt/c/ReMIND_work/paper/MIA/latex
sed -e 's/^\\documentclass\[review,authoryear\]{elsarticle}/\\documentclass[final,5p,twocolumn,authoryear]{elsarticle}\n\\usepackage{fontspec}\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]\\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]\n\\makeatletter\\def\\ps@pprintTitle{\\let\\@oddhead\\@empty\\let\\@evenhead\\@empty\\def\\@oddfoot{\\hfill\\thepage\\hfill}\\let\\@evenfoot\\@oddfoot}\\makeatother\n\\renewcommand{\\floatpagefraction}{0.8}\\renewcommand{\\dblfloatpagefraction}{0.8}\\renewcommand{\\topfraction}{0.95}\\renewcommand{\\dbltopfraction}{0.95}\\raggedbottom/' \
    -e 's/^\\linenumbers$//' -e '/^\\begin{highlights}/,/^\\end{highlights}/d' -e 's/^\\FloatBarrier$//' \
    -e 's/width=0.86\\textwidth\]{fig4_qualitative.pdf}/width=0.66\\textwidth]{fig4_qualitative.pdf}/' \
    -e 's/width=0.84\\textwidth\]{fig5_fields.pdf}/width=0.66\\textwidth]{fig5_fields.pdf}/' \
    -e 's/width=0.84\\textwidth\]{fig6_external.pdf}/width=0.64\\textwidth]{fig6_external.pdf}/' \
    -e 's/^\\begin{figure}\[p\]/\\begin{figure}[t]/' \
    -e 's/^\\begin{figure}\[\([!a-z]*\)\]/\\begin{figure*}[\1]/' -e 's/^\\end{figure}$/\\end{figure*}/' \
    -e 's/^\\begin{table}\[\([!a-z]*\)\]/\\begin{table*}[\1]/' -e 's/^\\end{table}$/\\end{table*}/' \
    main.tex > main_journal.tex
grep -c "figure\*\|table\*" main_journal.tex | sed 's/^/wide floats: /'
~/tectonic -X compile --keep-logs --keep-intermediates main_journal.tex > ../../build/tectonic_main_journal.out 2>&1 || true
~/tectonic -X compile --keep-logs main_journal.tex > ../../build/tectonic_main_journal.out 2>&1
rc=$?
mv -f main_journal.pdf ../pdf/main_journal.pdf
mv -f main_journal.log ../../build/main_journal.log 2>/dev/null || true
rm -f main_journal.aux main_journal.spl main_journal.xdv main_journal.bbl
echo "tectonic rc=$rc"
grep -a -o "Output written on [^(]*([0-9]* pages" ../../build/main_journal.log | tail -n 1
grep -a -c "Overfull" ../../build/main_journal.log | sed 's/^/overfull boxes: /'
grep -a "Overfull \\\\hbox ([0-9.]*pt" ../../build/main_journal.log | sed 's/.*(\([0-9.]*\)pt.*/\1/' | sort -rn | head -n 3 | sed 's/^/largest overfull (pt): /'
grep -a -c "undefined" ../../build/main_journal.log | sed 's/^/undefined: /'
