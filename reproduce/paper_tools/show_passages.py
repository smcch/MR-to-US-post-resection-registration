"""Print passages of the BUILT manuscript (text of the PDF, line numbers of the review layout removed), to read the
sentences with their numbers as the reader will see them.
Usage: python show_passages.py main|supplementary "start of passage" [length] ...   (pairs of key and optional length)
"""
import re
import sys
from pathlib import Path

doc = sys.argv[1]
t = (Path("/mnt/c/ReMIND_work/paper/build/pages") / f"{doc}.txt").read_text(encoding="utf-8")
t = re.sub(r"\n\s*\d{1,3}\s*(?=\n)", "", t)                          # margin line numbers
flat = re.sub(r"\s+", " ", t).replace("ﬃ", "ffi").replace("ﬁ", "fi").replace("ﬀ", "ff").replace("ﬂ", "fl")
args = sys.argv[2:]
i = 0
while i < len(args):
    key = args[i]
    n = 900
    if i + 1 < len(args) and args[i + 1].isdigit():
        n = int(args[i + 1])
        i += 1
    i += 1
    k = flat.find(key)
    print(f">> {key!r} at {k}")
    if k >= 0:
        print(flat[k: k + n])
    print()
