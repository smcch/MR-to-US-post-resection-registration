"""Abbreviation audit of the manuscript body (and of the supplement on its own): every all-caps token of 2-7 letters
(optionally with digits, such as ceT1, P70, SDLogJ, MIND-SSC) is listed with the line of its first use in the BODY
(after \\end{frontmatter}) and the line where a definition pattern "... (ABBR" or "ABBR (..." or "(ABBR; ..." first
appears; tokens that are used before being defined, or never defined, are flagged.  Known standard tokens are skipped.
Usage: python abbrev_audit.py [main|supplementary]"""
import re
import sys
from pathlib import Path

doc = sys.argv[1] if len(sys.argv) > 1 else "main"
p = Path("/mnt/c/ReMIND_work/paper/MIA/latex") / f"{doc}.tex"
lines = p.read_text(encoding="utf-8").splitlines()
start = next((i for i, l in enumerate(lines) if "\\end{frontmatter}" in l), 0) if doc == "main" else 0
SKIP = {"RESECT", "BITE", "ReMIND", "MCPO", "SyN", "ANTs", "NiftyReg", "SynthSeg", "GPU", "RTX", "GB", "PDF", "DOI", "LNCS", "IEEE",
        "MICCAI", "CPU", "RAS", "LPS", "NIfTI", "URL", "MIT", "ID", "CC", "BY", "SD", "CI", "CSF", "FLAIR", "T1", "T2", "ReMIND2Reg",
        "Learn2Reg", "CuRIOUS", "RaPTOR", "SSC", "MIND", "MR", "FE", "TRE", "iUS", "DoF", "EASY", "CRediT", "AI", "XXXX", "EOF",
        "ConvexAdam", "Adam", "MIND-SSC", "LNCC", "SDLogJ", "P70", "ceT1", "FoV", "TMI", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "S8", "S9", "S10", "S11", "S12", "HGG", "LGG", "IRB", "MRI", "US", "OK", "LTS", "MAD", "MI", "RMS", "NaN", "ANTsPy",
        "PyTorch", "SciPy", "FEM", "FER", "ICP", "CT", "ROI"}
tok = re.compile(r"(?<![A-Za-z\\])([A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9-]*)(?![A-Za-z])")
first, defined = {}, {}
for i, l in enumerate(lines[start:], start + 1):
    if l.lstrip().startswith("%"):
        continue
    for m in tok.finditer(l):
        t = m.group(1)
        if len(t) < 2 or len(t) > 9:
            continue
        first.setdefault(t, i)
        # definition patterns: "(TOKEN" , "(TOKEN;" , "TOKEN (" preceded by lowercase words, "TOKEN:" in a caption
        if re.search(r"\(" + re.escape(t) + r"[;,)]", l) or re.search(r"\(" + re.escape(t) + r"\s", l):
            defined.setdefault(t, i)
print(f"{doc}.tex: body from line {start + 1}")
for t, i in sorted(first.items(), key=lambda x: x[1]):
    d = defined.get(t)
    if d is None:
        print(f"  {t:12s} first use line {i:4d}  NOT DEFINED (pattern)" + ("   [standard]" if t in SKIP else ""))
    elif d > i:
        print(f"  {t:12s} first use line {i:4d}  defined later at line {d}" + ("   [standard]" if t in SKIP else ""))
