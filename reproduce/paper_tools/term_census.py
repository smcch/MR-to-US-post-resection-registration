"""Census of terminology variants in main.tex and supplementary.tex (counts and first lines), to drive the
unification pass.  Usage: python term_census.py"""
import re
from pathlib import Path

L = Path("/mnt/c/ReMIND_work/paper/MIA/latex")
GROUPS = {
    "proposed method": [r"FE refinement", r"\bthe refinement\b", r"proposed method", r"finite-element stage", r"deformable refinement stage",
                        r"\bthe stage\b", r"our pipeline", r"\bthe pipeline\b"],
    "steps": [r"modal fit", r"modal step", r"first step", r"modal field", r"modal displacement", r"local FE response", r"local response",
              r"\bFE response\b", r"second step", r"\bthe response\b"],
    "affine": [r"challenge affine", r"affine baseline", r"NiftyReg affine", r"classical baseline", r"refined affine",
               r"affine with FE refinement", r"affine with the FE refinement"],
    "pose": [r"MCPO pose", r"pose of MCPO", r"MCPO's pose", r"MCPO's own pose", r"refined pose\b", r"refined MCPO pose",
             r"leading challenge method", r"leading method", r"first-ranked method"],
    "init": [r"initialization", r"initial alignment", r"starting point"],
    "ius": [r"post-resection iUS", r"post-resection sweep", r"\bthe sweep\b", r"iUS sweep"],
    "tre": [r"case-mean TRE", r"patient-mean TRE", r"mean target registration error", r"\bcase means\b", r"per-case means"],
    "fov": [r"field of view", r"\bcone\b", r"iUS box", r"0\.5\\,mm box"],
    "external": [r"external site", r"external cohort", r"external patients", r"external-site patients", r"external clinical site"],
    "cohorts": [r"\bsources\b", r"landmark cohorts?", r"development cohorts", r"development data", r"public sets", r"public databases",
                r"held-out cohorts", r"evaluation sets?"],
    "descriptor": [r"MIND-SSC", r"MIND features", r"self-similarity descriptors", r"multimodal descriptor"],
    "ceT1": [r"ceT1", r"contrast-enhanced T1"],
    "ffd": [r"free-form stage", r"free-form B-spline refinement", r"B-spline free-form stage", r"NiftyReg free-form\b", r"free-form deformation"],
    "constants": [r"hyperparameter", r"\bconstants?\b", r"implementation constant", r"\bsettings?\b"],
    "floor": [r"annotation floor", r"noise floor", r"rigid bound", r"best-fit rigid"],
    "nav": [r"navigation alignment", r"navigation frame", r"navigation coordinates", r"navigation error", r"navigation-alignment"],
    "cavity": [r"resection cavity", r"resection site", r"\bcavity\b"],
    "corr": [r"block correspondences", r"image correspondences", r"image block correspondences"],
    "annot": [r"\bannotator\b", r"\bobserver\b", r"the author"],
    "hyphen": [r"preoperative-MR", r"preoperative MR to post-resection"],
    "units": [r"\bregistrations?\b", r"\bcases?\b", r"\bpatients?\b"],
    "deform-methods": [r"deformable methods", r"deformable stages"],
}
for name in ("main", "supplementary"):
    tex = (L / f"{name}.tex").read_text(encoding="utf-8")
    lines = tex.splitlines()
    print(f"===== {name}.tex")
    for g, pats in GROUPS.items():
        row = []
        for pat in pats:
            hits = [i + 1 for i, l in enumerate(lines) if re.search(pat, l) and not l.lstrip().startswith("%")]
            if hits:
                row.append(f"{pat.replace(chr(92), '')}: {len(hits)} ({', '.join(map(str, hits[:6]))}{', ...' if len(hits) > 6 else ''})")
        print(f"  [{g}] " + " | ".join(row))
