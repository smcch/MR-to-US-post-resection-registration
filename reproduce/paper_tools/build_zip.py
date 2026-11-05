"""Flat ZIP for Overleaf / the submission system: every LaTeX file and figure in one folder (main.tex compiles there
because \\graphicspath also lists the current directory)."""
import zipfile
from pathlib import Path

P = Path("/mnt/c/ReMIND_work/paper/MIA")
out = P / "FE_refinement_Overleaf.zip"
files = sorted(P.glob("latex/*.tex")) + sorted(P.glob("imagenes/fig*.pdf"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.name)
print(out, len(files), "files", f"{out.stat().st_size / 1e6:.1f} MB")
