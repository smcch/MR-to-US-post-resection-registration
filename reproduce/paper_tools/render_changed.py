"""Print every unique sentence of main.tex / supplementary.tex that uses a macro whose value changed between two
numbers.tex files, with the macros expanded to NEW [old] values, to re-read the wording after a re-scoring.
Usage: python render_changed.py OLD/numbers.tex NEW/numbers.tex TEXDIR > out.txt"""
import re
import sys
from pathlib import Path

pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def load(p):
    d = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


A, B = load(sys.argv[1]), load(sys.argv[2])
changed = {k for k in set(A) | set(B) if A.get(k) != B.get(k)}
mac = re.compile(r"\\([A-Za-z]+)(\{\})?")
for f in ("main.tex", "supplementary.tex"):
    s = Path(sys.argv[3], f).read_text(encoding="utf-8")
    s = re.sub(r"(?m)^%.*$", "", s)
    paras = re.split(r"\n\s*\n", s)
    print(f"=================== {f}")
    for p in paras:
        sents = re.split(r"(?<=[.;])\s+(?=[A-Z\\(])", " ".join(p.split()))
        for x in sents:
            names = [m.group(1) for m in mac.finditer(x)]
            if not any(n in changed for n in names):
                continue

            def rep(m):
                n = m.group(1)
                if n in B:
                    return f"{B[n]}[{A.get(n, '-')}]" if n in changed else B[n]
                return m.group(0)
            print("- " + mac.sub(rep, x)[:900])
