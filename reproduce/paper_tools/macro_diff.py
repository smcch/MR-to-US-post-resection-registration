"""Compare two numbers.tex files and show, for every macro whose value changed, the sentences of main.tex and
supplementary.tex that use it (to re-read qualitative wording after a re-scoring).
Usage: python macro_diff.py OLD/numbers.tex NEW/numbers.tex [--tex MIA/latex] [--only-used]"""
import re
import sys
from pathlib import Path

old, new = Path(sys.argv[1]), Path(sys.argv[2])
tex = Path(sys.argv[sys.argv.index("--tex") + 1]) if "--tex" in sys.argv else new.parent
pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def load(p):
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


A, B = load(old), load(new)
changed = {k: (A.get(k), B.get(k)) for k in set(A) | set(B) if A.get(k) != B.get(k)}
docs = {f: (tex / f).read_text(encoding="utf-8") for f in ("main.tex", "supplementary.tex")}
used = {}
for f, s in docs.items():
    sents = re.split(r"(?<=[.;])\s+", s)
    for k in changed:
        hits = [x for x in sents if re.search(r"\\" + k + r"(?![A-Za-z])", x)]
        if hits:
            used.setdefault(k, []).extend((f, " ".join(h.split())[:400]) for h in hits)
print(f"{len(changed)} macros changed, {len(used)} of them used in the text")
for k in sorted(used):
    o, n = changed[k]
    print(f"\n\\{k}: {o} -> {n}")
    for f, h in used[k]:
        print(f"   [{f}] {h}")
if "--only-used" not in sys.argv:
    print("\nchanged but not used in the text:", ", ".join(sorted(set(changed) - set(used)))[:3000])
