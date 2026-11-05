"""Checks of the built manuscript, read from the LaTeX logs and sources in Python (a shell grep hid problems before):
pages, overfull boxes, undefined references / citations / macros, missing files; journal limits of the front matter
(abstract <= 250 words, highlights <= 85 characters, keywords <= 6) with the number macros expanded; body word count.
"""
import re
from pathlib import Path

P = Path("/mnt/c/ReMIND_work/paper")
TEX = P / "MIA/latex"
BUILD = P / "build"
ok = True


def log_report(doc):
    global ok
    lp = BUILD / f"{doc}.log"
    if not lp.exists():
        print(f"[{doc}] NO LOG")
        ok = False
        return
    log = lp.read_text(errors="replace")
    pages = re.findall(r"Output written on .*?\((\d+) pages?", log)
    over = [(float(m.group(2)), m.group(1)) for m in re.finditer(r"Overfull \\([hv])box \(([\d.]+)pt too (?:wide|high)\)", log)]
    big = [o for o in over if o[0] > 5.0]
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)'.*?undefined", log)))
    errors = re.findall(r"^! (.*)$", log, flags=re.M)
    missing = re.findall(r"File `([^']+)' not found", log)
    print(f"[{doc}] pages {pages[-1] if pages else '?'}; overfull boxes {len(over)} (> 5 pt: {len(big)}"
          + (", largest %.1f pt" % max(o[0] for o in over) if over else "") + f"); undefined refs/cites {undef}; errors {errors[:5]}; "
          f"missing files {missing}")
    out = (BUILD / f"tectonic_{doc}.out").read_text(errors="replace") if (BUILD / f"tectonic_{doc}.out").exists() else ""
    terr = [l for l in out.splitlines() if l.startswith("error")]
    if terr:
        print(f"[{doc}] tectonic errors: {terr[:5]}")
    if errors or undef or missing or terr or not pages:
        ok = False
    if big:
        ctx = re.findall(r"(Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*\n[^\n]*)", log)
        for c in ctx:
            if float(c[1]) > 5.0:
                print("   ", c[0].replace("\n", " | ")[:200])


def macros():
    m = {}
    for line in (TEX / "numbers.tex").read_text().splitlines():
        g = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if g:
            m[g.group(1)] = g.group(2)
    return m


def expand(s, m):
    for _ in range(3):
        s = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda g: m[g.group(1)] if g.group(1) in m else g.group(0), s)
    return s


def plain(s):
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:emph|textbf|texttt|mathrm|mathsf)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", " ").replace("~", " ").replace("\\%", "%").replace("$", "").replace("\\ ", " ")
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def front_matter():
    global ok
    src = (TEX / "main.tex").read_text(encoding="utf-8")
    m = macros()
    ab = plain(expand(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, flags=re.S).group(1), m))
    n = len(ab.split())
    print(f"[front] abstract {n} words (limit 250){'' if n <= 250 else '  <-- TOO LONG'}")
    ok &= n <= 250
    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", src, flags=re.S).group(1)
    items = [plain(expand(x, m)) for x in hl.split("\\item")[1:]]
    for it in items:
        print(f"[front] highlight {len(it):3d} chars{'' if len(it) <= 85 else '  <-- TOO LONG'}: {it}")
        ok &= len(it) <= 85
    print(f"[front] highlights {len(items)} (3 to 5)")
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, flags=re.S).group(1).split("\\sep")
    print(f"[front] keywords {len(kw)} (max 6)")
    body = src[src.index("\\section{Introduction}"):src.index("\\section*{Declarations}")]
    body = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", " ", body, flags=re.S)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    print(f"[front] body {len(plain(expand(body, m)).split())} words (previous manuscript version: about 5400)")
    left = sorted(set(re.findall(r"\\((?:Rpo|Bi|Rpr|Re|Ex|Pub|Rs|All|Pat|Aff|Time|Mem|Mc|Model|Holm|Corr|Sens|Fig)[A-Z][A-Za-z]+)", src)) - set(m))
    if left:
        print(f"[front] macros used but not defined in numbers.tex: {left}")
        ok = False


for d in ("main", "supplementary"):
    log_report(d)
front_matter()
print("CHECK", "OK" if ok else "PROBLEMS")
