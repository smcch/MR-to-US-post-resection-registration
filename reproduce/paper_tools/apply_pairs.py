"""Apply (old, new) replacement pairs produced by the style editors to a LaTeX file, safely:
each `old` must occur EXACTLY once in the current file; pairs that do not match (or match more than once) are skipped
and listed, so that they can be applied by hand.  A backup of the target is written next to it (<name>.bak_<tag>).
Usage: python apply_pairs.py <target.tex> <pairs.py> [<pairs.py> ...]
The pairs file is a Python file defining PAIRS = [(old, new), ...]."""
import importlib.util
import shutil
import sys
import time
from pathlib import Path

target = Path(sys.argv[1])
s = target.read_text(encoding="utf-8")
tag = time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(target, target.with_name(target.name + f".bak_{tag}"))
applied, skipped = 0, []
var = "PAIRS"                      # "--var NAME" before a pairs file selects another list of that file (e.g. PAIRS_SUPP)
args = sys.argv[2:]
while args:
    pf = args.pop(0)
    if pf == "--var":
        var = args.pop(0)
        continue
    spec = importlib.util.spec_from_file_location("pairs", pf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for i, (old, new) in enumerate(getattr(mod, var)):
        n = s.count(old)
        if n == 1:
            s = s.replace(old, new, 1)
            applied += 1
        else:
            # second chance: normalise line breaks inside the pair (editors sometimes re-wrap)
            o2 = " ".join(old.split())
            s2 = " ".join(s.split())
            if n == 0 and s2.count(o2) == 1:
                # locate by collapsing whitespace: rebuild a regex that allows any whitespace run
                import re
                pat = r"\s+".join(re.escape(w) for w in old.split())
                m = list(re.finditer(pat, s))
                if len(m) == 1:
                    s = s[:m[0].start()] + new + s[m[0].end():]
                    applied += 1
                    continue
            skipped.append((Path(pf).name, i, n, old[:90].replace("\n", " ")))
target.write_text(s, encoding="utf-8")
print(f"{target.name}: {applied} pairs applied, {len(skipped)} skipped (backup {target.name}.bak_{tag})")
for pf, i, n, head in skipped:
    print(f"  SKIPPED {pf}[{i}] (matches {n}): {head}")
