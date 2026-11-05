"""Word count per section of main.tex (macros expanded with numbers.tex), for two versions: current and a backup.
Usage: python _section_words.py <backup.tex>"""
import re
import sys
from pathlib import Path

L = Path("/mnt/c/ReMIND_work/paper/MIA/latex")
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (L / "numbers.tex").read_text(encoding="utf-8")))


def words(tex):
    tex = re.sub(r"(?m)^%.*$", "", tex)
    tex = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", tex, flags=re.S)       # floats out
    tex = re.sub(r"\\(\w+)\{\}", lambda m: mac.get(m.group(1), m.group(0)), tex)
    tex = re.sub(r"\\(\w+)(?![A-Za-z])", lambda m: mac.get(m.group(1), " "), tex)
    tex = re.sub(r"\$[^$]*\$", " x ", tex)
    tex = re.sub(r"[{}\\~]", " ", tex)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%'-]*", tex))


def sections(tex):
    body = tex[tex.index("\\end{frontmatter}"):tex.index("\\section*{Declarations}")]
    parts = re.split(r"(?m)^\\(?:sub)?section\{([^}]*)\}", body)
    out = {}
    for i in range(1, len(parts), 2):
        out[parts[i]] = words(parts[i + 1])
    return out


cur = sections((L / "main.tex").read_text(encoding="utf-8"))
old = sections(Path(sys.argv[1]).read_text(encoding="utf-8"))
tot_c = tot_o = 0
for k in cur:
    o = old.get(k, old.get(next((kk for kk in old if kk[:12] == k[:12]), ""), 0))
    tot_c += cur[k]; tot_o += o
    print(f"{k[:58]:58s} {o:5d} -> {cur[k]:5d}  ({cur[k] - o:+d})")
print(f"{'TOTAL':58s} {tot_o:5d} -> {tot_c:5d}  ({tot_c - tot_o:+d})")
