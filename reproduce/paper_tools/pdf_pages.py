"""Render PDF pages to PNG (visual check of the built manuscript) and dump the text.
Usage: python pdf_pages.py main|supplementary [dpi]   ->  paper/build/pages/<doc>_pNN.png, <doc>.txt
"""
import sys
from pathlib import Path

import fitz

P = Path("/mnt/c/ReMIND_work/paper")
doc = sys.argv[1]
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 70
out = P / "build/pages"
out.mkdir(parents=True, exist_ok=True)
pdf = fitz.open(str(P / f"MIA/pdf/{doc}.pdf"))
text = []
for i, page in enumerate(pdf):
    page.get_pixmap(dpi=dpi).save(str(out / f"{doc}_p{i + 1:02d}.png"))
    text.append(f"\n===== page {i + 1} =====\n" + page.get_text())
(out / f"{doc}.txt").write_text("".join(text), encoding="utf-8")
print(doc, len(pdf), "pages ->", out)
