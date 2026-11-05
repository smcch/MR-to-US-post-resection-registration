"""American English spelling for the manuscript (author's choice, 2026-10-03): applies a fixed word list to the LaTeX
sources, the table strings of build_numbers.py and the labels of the figure scripts.  Whole words only, case
preserved for capitalised forms; macro names (\\AffNotReproTre, ...) are not words and are left alone.
Usage: python americanize.py            (idempotent; prints the replacements made per file)
"""
import re
from pathlib import Path

P = Path("/mnt/c/ReMIND_work/paper")
WORDS = {
    "initialisation": "initialization", "initialisations": "initializations", "initialised": "initialized", "initialise": "initialize",
    "initialising": "initializing", "tumour": "tumor", "tumours": "tumors", "normalised": "normalized", "normalise": "normalize",
    "normalisation": "normalization", "colour": "color", "colours": "colors", "coloured": "colored", "millimetre": "millimeter",
    "millimetres": "millimeters", "grey": "gray", "organisers": "organizers", "organiser": "organizer", "centre": "center",
    "centres": "centers", "centred": "centered", "afterwards": "afterward", "towards": "toward", "relabelled": "relabeled",
    "relabelling": "relabeling", "labelled": "labeled", "labelling": "labeling", "regulariser": "regularizer",
    "regularisation": "regularization", "regularised": "regularized", "regularise": "regularize", "summarises": "summarizes",
    "summarise": "summarize", "summarised": "summarized", "optimises": "optimizes", "optimised": "optimized",
    "optimisation": "optimization", "optimise": "optimize", "optimiser": "optimizer", "modelled": "modeled", "modelling": "modeling",
    "localisation": "localization", "labour": "labor", "haemostatic": "hemostatic", "generalised": "generalized",
    "generalisation": "generalization", "behaviour": "behavior", "artefact": "artifact", "artefacts": "artifacts",
    "anonymised": "anonymized", "analysed": "analyzed", "analyse": "analyze", "analysing": "analyzing", "minimise": "minimize",
    "minimises": "minimizes", "minimised": "minimized", "minimisation": "minimization", "maximise": "maximize",
    "characterise": "characterize", "characterised": "characterized", "emphasise": "emphasize", "recognise": "recognize",
    "recognisable": "recognizable", "realise": "realize", "penalise": "penalize", "penalised": "penalized",
    "discretisation": "discretization", "discretised": "discretized", "parametrised": "parameterized", "visualisation": "visualization",
    "stabilise": "stabilize", "utilise": "utilize", "specialised": "specialized", "synthesised": "synthesized", "standardised": "standardized",
    "randomised": "randomized", "quantisation": "quantization", "rigour": "rigor", "favour": "favor", "neighbour": "neighbor",
    "neighbouring": "neighboring", "whilst": "while", "learnt": "learned", "fibre": "fiber", "fibres": "fibers", "metre": "meter",
    "theatre": "theater", "judgement": "judgment", "acknowledgement": "acknowledgment", "oedema": "edema", "haemorrhage": "hemorrhage",
    "anaesthesia": "anesthesia", "paediatric": "pediatric", "orientated": "oriented", "upwards": "upward", "backwards": "backward",
    "forwards": "forward", "catalogue": "catalog", "analogue": "analog", "programme": "program", "practise": "practice",
    "fulfil": "fulfill", "enrol": "enroll", "skilful": "skillful", "defence": "defense", "licence": "license", "ageing": "aging",
}
FILES = [P / "MIA/latex/main.tex", P / "MIA/latex/supplementary.tex"]
SCRIPT_ONLY_STRINGS = [P / "tools/build_numbers.py", P / "tools/fig_pipeline.py",
         P / "tools/fig_model.py", P / "tools/fig_method_lib.py", P / "tools/fig_cases_stats.py", P / "tools/fig_qualitative.py",
         P / "tools/fig_fields.py", P / "tools/fig_lib.py"]   # scripts: labels fixed by hand (identifiers such as centres_mm must not change)
pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, WORDS), key=len, reverse=True)) + r")\b", flags=re.IGNORECASE)


def repl(m):
    w = m.group(0)
    low = w.lower()
    if low not in WORDS:
        return w
    out = WORDS[low]
    if w.isupper():
        return out.upper()
    if w[0].isupper():
        return out[0].upper() + out[1:]
    return out


for f in FILES:
    s = f.read_text(encoding="utf-8")
    n = len(pat.findall(s))
    t = pat.sub(repl, s)
    if t != s:
        f.write_text(t, encoding="utf-8")
    print(f"{f.name}: {n} replacement(s)")
