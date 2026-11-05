"""Every number of the FE-refinement manuscript, from the per-case and per-landmark tables.

Reads   /mnt/c/ReMIND_work/fer/tables/{PER_CASE.csv,LANDMARKS.csv}   (scripts/fer_score.py)
        /mnt/c/ReMIND_work/fer/tables/AFFINE_TIMING.csv              (scripts/fer_time_affine.py, optional)
Writes  paper/MIA/latex/numbers.tex          macros used by main.tex and supplementary.tex
        paper/MIA/latex/tab_*.tex, supp_*.tex tables
        paper/tools/NUMBERS.md               macro -> value -> definition (traceability)
        paper/build/stats.json               everything, unrounded
Nothing is typed by hand in the manuscript: when the landmark files change, rerun fer_score.py and this script.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("/mnt/c/ReMIND_work")
TAB = W / "fer/tables"
TEX = W / "paper/MIA/latex"
TOOLS = W / "paper/tools"
BUILD = W / "paper/build"
sys.path.insert(0, str(W / "scripts"))
import fer_common as C                                               # noqa

COH = {"resect_after": "Rpo", "bite_g4": "Bi", "resect_before": "Rpr", "remind": "Re", "external": "Ex"}
ARM = {"initial": "Ini", "affine": "Aff", "f3d_mind": "Ffd", "convexadam": "Ca", "deeds": "De", "syn": "Sy", "fer": "Fer",
       "mcpo": "Mc", "mcpo_pose": "Mp", "mcpo_pose_fer": "Mpf", "bestfit": "Bf"}
LM_COH = ["resect_after", "bite_g4", "resect_before", "remind"]            # cohorts with paired tests
POST_LM = ["resect_after", "bite_g4", "remind", "external"]                # post-resection cohorts with paired tests
MAC, DOC = {}, []


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def load():
    rows = []
    for r in csv.DictReader(open(TAB / "PER_CASE.csv")):
        d = {k: (v if k in ("cohort", "case", "mcpo_moving") else num(v)) for k, v in r.items()}
        rows.append(d)
    lms = list(csv.DictReader(open(TAB / "LANDMARKS.csv")))
    return rows, lms


def put(name, text, definition, raw=None):
    assert name.isalpha(), name
    assert name not in MAC, f"duplicate macro {name}"
    MAC[name] = str(text)
    DOC.append((name, str(text), definition, raw))


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def ffold(x, allzero):
    """folding percentage: 0 when no voxel folds, '<0.01' below the resolution of the table, one decimal above 1 %"""
    if allzero:
        return "0"
    if x < 0.005:
        return "$<$0.01"
    return f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def fs(x):
    """two decimals with a typographic minus sign"""
    return f"{x:.2f}".replace("-", "$-$")


def pfmt(p):
    """(bare, with relation)"""
    if not np.isfinite(p):
        return "n/a", "n/a"
    if p < 0.001:
        return "$<$0.001", "$<$\\,0.001"
    s = f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    return s, f"=\\,{s}"


def wil(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, y):
        return np.nan
    return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def clean_floor(us, mr, k_mad=3.0):
    """lm_select.analyse: mean residual of the best-fit rigid after iteratively dropping pairs beyond median + 3 MAD."""
    keep = np.ones(len(us), bool)
    for _ in range(5):
        R, t = C.horn(us[keep], mr[keep])
        r = np.linalg.norm(us @ R.T + t - mr, axis=1)
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med)) * 1.4826 + 1e-9
        new = r <= med + k_mad * mad
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    R, t = C.horn(us[keep], mr[keep])
    r = np.linalg.norm(us @ R.T + t - mr, axis=1)
    return float(r[keep].mean()), int((~keep).sum())


def main():
    rows, lms = load()
    S = {}                                                           # unrounded statistics for stats.json
    G = {p: [r for r in rows if r["cohort"] == c] for c, p in COH.items()}
    G["Pub"] = [r for r in rows if r["cohort"] in C.PUBLIC]
    G["Rs"] = [r for r in rows if r["cohort"] in ("resect_after", "resect_before")]
    G["All"] = list(rows)
    G["Post"] = [r for r in rows if r["cohort"] != "resect_before"]
    G["Pp"] = [r for r in rows if r["cohort"] in ("resect_after", "bite_g4")]
    names = {"Rpo": "RESECT post-resection", "Bi": "BITE post-resection", "Rpr": "RESECT pre-resection", "Re": "ReMIND",
             "Ex": "external site", "Pub": "RESECT + BITE (public sets, both regimes)", "Rs": "RESECT (both sets)",
             "All": "all registrations (both regimes)", "Post": "post-resection registrations (RESECT post, BITE, ReMIND, external)",
             "Pp": "RESECT post-resection + BITE (public post-resection sets)"}

    def col(rr, k):
        return np.array([r.get(k, np.nan) for r in rr], float)

    # ---------------------------------------------------------------- sizes
    for g, rr in G.items():
        put(f"{g}N", len(rr), f"number of registrations, {names[g]}")
    for c, p in COH.items():
        put(f"{p}Pairs", sum(1 for x in lms if x["cohort"] == c), f"landmark pairs, {names[p]}")
    put("AllPairs", len(lms), "landmark pairs, all sources")
    # two-observer annotation of ReMIND and the external site (scripts/landmarks_final_build.py -> STATS.json)
    import json as _js
    stp = W / "landmarks_final/STATS.json"
    if stp.exists():
        st = _js.loads(stp.read_text())
        put("LmDupMm", f"{st['dup_mm']:.0f}", "MR points closer than this (mm) are the same feature (merged)")
        put("LmOutMm", f2(st["outlier_mm"]), "an iUS pick farther than this (mm) from both other picks is an outlier")
        for key, p in (("ReMIND", "Re"), ("External", "Ex"), ("Pooled", "Two")):
            s = st[key]
            for k, mk in (("annotated", "Annot"), ("merged", "Merged"), ("excluded", "Excl"), ("kept", "Kept"),
                          ("kept_min", "KeptMin"), ("kept_max", "KeptMax")):
                put(f"{p}Lm{mk}", s[k], f"landmark pairs {k.replace('_', ' ')}, {key} (two-observer annotation)")
            for k, mk in (("intra", "Intra"), ("inter_mean", "InterMean"), ("inter_one", "InterOne"), ("inter_two", "InterTwo")):
                for mode, mm in (("final", ""), ("all", "All")):
                    put(f"{p}Obs{mk}{mm}M", f2(s[f'{k}_{mode}_mean']), f"{k} distance of iUS picks, {mode} set, mean of case means (mm), {key}")
                    put(f"{p}Obs{mk}{mm}Sd", f2(s[f'{k}_{mode}_sd']), "SD over cases of the same (mm)")

    # ---------------------------------------------------------------- arm summaries
    for g, rr in G.items():
        for a, ak in ARM.items():
            v = col(rr, a)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            S[f"{g}.{a}"] = dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, median=np.median(v),
                                 min=v.min(), max=v.max())
            put(f"{g}{ak}M", f2(v.mean()), f"mean of case-mean TRE (mm), {a}, {names[g]} (n {len(v)})", float(v.mean()))
            if len(v) > 1:
                put(f"{g}{ak}S", f2(v.std(ddof=1)), f"SD over cases, {a}, {names[g]}")
                put(f"{g}{ak}Md", f2(np.median(v)), f"median over cases, {a}, {names[g]}")
                put(f"{g}{ak}Lo", f2(v.min()), f"minimum case mean, {a}, {names[g]}")
                put(f"{g}{ak}Hi", f2(v.max()), f"maximum case mean, {a}, {names[g]}")
                put(f"{g}{ak}Ok", f"{100 * (v <= 3).mean():.0f}", f"% of cases with mean TRE <= 3 mm, {a}, {names[g]}")
                put(f"{g}{ak}Bad", int((v > 5).sum()), f"cases with mean TRE > 5 mm, {a}, {names[g]}")
            t = col(rr, f"{a}_tre30")
            t = t[np.isfinite(t)]
            if len(t):
                put(f"{g}{ak}T", f2(t.mean()), f"TRE30 (70th percentile per case, mean over cases), {a}, {names[g]}")
        for k, mk in (("f3d_lncc", "FfdL"), ("f3d_mind_lncc", "FfdML")):
            v = col(rr, k)
            v = v[np.isfinite(v)]
            if len(v):
                put(f"{g}{mk}M", f2(v.mean()), f"mean TRE, {k} on the challenge affine, {names[g]} (n {len(v)})")
        put(f"{g}McN", int(np.isfinite(col(rr, "mcpo")).sum()), f"registrations with an MCPO result, {names[g]}")
        put(f"{g}MpN", int(np.isfinite(col(rr, "mcpo_pose")).sum()), f"registrations with an MCPO pose, {names[g]}")
        if g == "Re":
            own = np.nan_to_num(col(rr, "mcpo_own_box")) > 0          # ReMIND-009: MCPO on our box (not in the release)
            put("ReRelN", int((np.isfinite(col(rr, "mcpo")) & ~own).sum()), "ReMIND patients whose MCPO ran on the challenge's own files")
            put("ReOwnN", int(own.sum()), "ReMIND patients whose MCPO ran on our 0.5 mm box (not in the release)")
            put("ReOwnCases", ", ".join(r["case"] for r, o in zip(rr, own) if o), "those patients")

    # ---------------------------------------------------------------- paired contrasts
    PAIRS = [("fer", "affine"), ("fer", "f3d_mind"), ("fer", "mcpo"), ("fer", "initial"), ("mcpo", "affine"), ("f3d_mind", "affine"),
             ("mcpo", "initial"), ("affine", "initial"), ("mcpo_pose_fer", "mcpo_pose"), ("mcpo_pose_fer", "mcpo"), ("mcpo_pose_fer", "fer"),
             ("mcpo_pose", "affine"), ("mcpo", "mcpo_pose"), ("fer", "mcpo_pose"),
             ("fer", "convexadam"), ("fer", "deeds"), ("fer", "syn"), ("convexadam", "affine"), ("deeds", "affine"), ("syn", "affine"),
             ("convexadam", "mcpo")]
    praw = {}
    for g, rr in G.items():
        for x, y in PAIRS:
            a, b = col(rr, x), col(rr, y)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if len(a) == 0:
                continue
            key = f"{g}{ARM[x]}v{ARM[y]}"
            d = b - a                                                # positive = x better
            p = wil(a, b)
            praw[(g, x, y)] = p
            S[key] = dict(n=len(a), mean_x=a.mean(), mean_y=b.mean(), diff=d.mean(), better=int((a < b).sum()), p=p,
                          worst=float((a - b).max()))
            put(f"{key}D", f2(d.mean()), f"mean of ({y} - {x}) over cases (positive = {x} better), {names[g]} (n {len(a)})", float(d.mean()))
            put(f"{key}N", int((a < b).sum()), f"cases in which {x} < {y}, {names[g]} (of {len(a)})")
            put(f"{key}NofT", f"{int((a < b).sum())} of {len(a)}", f"cases in which {x} < {y}, {names[g]}")
            put(f"{key}Worse", int((a > b).sum()), f"cases in which {x} > {y}, {names[g]} (of {len(a)})")
            put(f"{key}WorseMax", f2(max(0.0, float((a - b).max()))), f"largest ({x} - {y}) over cases, {names[g]}")
            put(f"{key}Lo", f2(float(d.min())), f"smallest ({y} - {x}), {names[g]}")
            put(f"{key}Hi", f2(float(d.max())), f"largest ({y} - {x}), {names[g]}")
            if len(a) > 1:
                pb, pr = pfmt(p)
                put(f"{key}P", pb, f"Wilcoxon signed-rank (zsplit) {x} vs {y}, {names[g]}, unadjusted", p)
                put(f"{key}Pr", pr, "same, with its relation sign")
                lo, hi = boot_ci(d)
                put(f"{key}CiLo", fs(lo), f"bootstrap 95% CI of mean ({y} - {x}), {names[g]}")
                put(f"{key}CiHi", fs(hi), "upper limit")
    fam = [(COH[c], x, y) for (x, y) in (("fer", "affine"), ("mcpo_pose_fer", "mcpo_pose")) for c in POST_LM]
    adj = holm([praw[k] for k in fam])
    for (g, x, y), pa in zip(fam, adj):
        pb, pr = pfmt(pa)
        put(f"{g}{ARM[x]}v{ARM[y]}Ph", pb, f"Holm-adjusted p (primary family of {len(fam)}: refinement vs its initialization, affine and MCPO pose, "
            f"in the 4 post-resection landmark cohorts)", float(pa))
        put(f"{g}{ARM[x]}v{ARM[y]}Phr", pr, "same, with its relation sign")
        S[f"{g}{ARM[x]}v{ARM[y]}"]["p_holm"] = float(pa)
    put("HolmFamily", len(fam), "size of the primary family")
    put("HolmMax", pfmt(max(adj))[0], "largest Holm-adjusted p in the primary family")

    # ---------------------------------------------------------------- patient-level pooling
    def pid(r):
        return ("RESECT-" + r["case"]) if r["cohort"].startswith("resect") else (r["cohort"] + "-" + r["case"])
    for x, y, tag in (("fer", "affine", "PatFervAff"), ("fer", "f3d_mind", "PatFervFfd"), ("fer", "mcpo", "PatFervMc")):
        byp = {}
        for r in rows:
            if np.isfinite(r.get(x, np.nan)) and np.isfinite(r.get(y, np.nan)):
                byp.setdefault(pid(r), []).append(r[y] - r[x])
        d = np.array([np.mean(v) for v in byp.values()])
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
        lo, hi = boot_ci(d)
        S[tag] = dict(n=len(d), diff=d.mean(), better=int((d > 0).sum()), p=p, ci=(lo, hi))
        if tag == "PatFervAff":
            put("PatN", len(d), "patients (RESECT patients counted once across their two sets)")
        put(f"{tag}Units", len(d), f"patients with both {x} and {y}")
        put(f"{tag}D", fs(d.mean()), f"patient-level mean of ({y} - {x})", float(d.mean()))
        put(f"{tag}CiLo", fs(lo), "bootstrap 95% CI, lower")
        put(f"{tag}CiHi", fs(hi), "bootstrap 95% CI, upper")
        if tag == "PatFervMc":                                       # the same contrast seen from MCPO's side
            put("PatMcvFerD", fs(-d.mean()), "patient-level mean of (FER - MCPO): positive = MCPO better")
            put("PatMcvFerCiLo", fs(-hi), "bootstrap 95% CI of (FER - MCPO), lower")
            put("PatMcvFerCiHi", fs(-lo), "bootstrap 95% CI of (FER - MCPO), upper")
        put(f"{tag}N", f"{int((d > 0).sum())} of {len(d)}", f"patients in which {x} < {y}")
        pb, pr = pfmt(p)
        put(f"{tag}P", pb, f"Wilcoxon signed-rank on patient-level differences, {x} vs {y}", p)
        put(f"{tag}Pr", pr, "same, with its relation sign")

    # ---------------------------------------------------------------- harm, failures of the initialization
    pub, allr = G["Pub"], G["All"]
    w = col(pub, "affine") - col(pub, "initial")
    put("PubAffWorseNavN", int((w > 0).sum()), "public registrations in which the affine ends above the navigation alignment")
    put("PubAffWorseNavMax", f2(w.max()), "largest (affine - initial), public sets")
    put("PubFailChange", f"{np.mean((col(pub, 'fer') - col(pub, 'affine'))[w > 0]):+.2f}".replace("-", "$-$").replace("+", "$+$"),
        "mean (FER - affine) on those registrations")
    put("PubFailGain", f2(np.mean((col(pub, "affine") - col(pub, "fer"))[w > 0])), "mean (affine - FER) on those registrations")
    put("PubFailRemainN", int(((col(pub, "fer") > col(pub, "initial")) & (w > 0)).sum()),
        "of those registrations, how many remain above the navigation alignment after the refinement")
    w2 = col(pub, "fer") - col(pub, "initial")
    put("PubFerWorseNavN", int((w2 > 0).sum()), "public registrations in which affine + FER ends above the navigation alignment")
    put("PubFerWorseNavMax", f2(w2.max()), "largest (FER - initial), public sets")
    wr = col(G["Re"], "affine") - col(G["Re"], "initial")
    put("ReAffWorseNavN", int((wr > 0).sum()), "ReMIND patients in which the affine ends above the navigation alignment")
    for g in ("Rpo", "Bi", "Rpr", "Re"):
        a, i = col(G[g], "affine"), col(G[g], "initial")
        put(f"{g}AffGain", f"{100 * (1 - a.mean() / i.mean()):.0f}", f"% of the initial mean error removed by the affine, {names[g]}")
    gains = [100 * (1 - col(G[g], 'affine').mean() / col(G[g], 'initial').mean()) for g in ("Rpo", "Bi", "Rpr", "Re")]
    put("AffGainLo", f"{min(gains):.0f}", "smallest % of the initial error removed by the affine over the four cohorts")
    put("AffGainHi", f"{max(gains):.0f}", "largest")
    put("AffScaleLo", f2(min(col(allr, "aff_scale_min"))), "smallest singular value of the affines")
    put("AffScaleHi", f2(max(col(allr, "aff_scale_max"))), "largest singular value of the affines")
    out = (col(allr, "aff_scale_min") < 0.9) | (col(allr, "aff_scale_max") > 1.1)
    put("AffScaleOutN", int(out.sum()), "registrations with a scale factor outside 0.90-1.10")
    # RESECT: registrations in which the affine leaves more than 5 mm
    rs = G["Rs"]
    bad = col(rs, "affine") > 5
    put("RsAffFailN", int(bad.sum()), "RESECT registrations (both sets) in which the affine leaves > 5 mm")
    put("RsAffFailLo", f1(col(rs, "affine")[bad].min()), "smallest affine error among them")
    put("RsAffFailHi", f1(col(rs, "affine")[bad].max()), "largest")
    put("RsAffFailRotLo", f"{col(rs, 'aff_rot_deg')[bad].min():.0f}", "smallest rotation of the affine among them (deg)")
    put("RsAffFailRotHi", f"{col(rs, 'aff_rot_deg')[bad].max():.0f}", "largest")
    put("RsAffFailMcLo", f1(col(rs, "mcpo")[bad].min()), "MCPO error on those registrations, smallest")
    put("RsAffFailMcHi", f1(col(rs, "mcpo")[bad].max()), "largest")
    put("RsAffFailFerM", f2(col(rs, "fer")[bad].mean()), "FER mean on those registrations")
    put("RsOkN", int((~bad).sum()), "the other RESECT registrations")
    put("RsOkFerM", f2(col(rs, "fer")[~bad].mean()), "FER mean on the other RESECT registrations")
    put("RsOkMcM", f2(col(rs, "mcpo")[~bad].mean()), "MCPO mean on the other RESECT registrations")
    put("RsOkAffM", f2(col(rs, "affine")[~bad].mean()), "affine mean on the other RESECT registrations")
    put("RsOkFervMcP", pfmt(wil(col(rs, "fer")[~bad], col(rs, "mcpo")[~bad]))[0], "Wilcoxon FER vs MCPO on the other RESECT registrations")
    put("RsOkFervMcPr", pfmt(wil(col(rs, "fer")[~bad], col(rs, "mcpo")[~bad]))[1], "same, with its relation sign")
    put("RsOkGap", f2(col(rs, "fer")[~bad].mean() - col(rs, "mcpo")[~bad].mean()), "FER minus MCPO on the other RESECT registrations")
    put("RsOkFerBetterN", f"{int((col(rs, 'fer')[~bad] < col(rs, 'mcpo')[~bad]).sum())} of {int((~bad).sum())}", "of them, FER < MCPO")
    S["RsFail"] = dict(cases=[(r["cohort"], r["case"], r["affine"], r["aff_rot_deg"], r["fer"], r["mcpo"]) for r, b in zip(rs, bad) if b])
    big = col(allr, "aff_rot_deg") > 10
    put("AllAffRotBigN", int(big.sum()), "registrations with an affine rotation above 10 deg")

    # ---------------------------------------------------------------- regularity, size of the refinement
    for g in ("Pub", "Re", "All", "Ex", "Pp", "Post", "Rpo", "Bi", "Rpr"):
        rr = G[g]
        for key, mk in (("fer", "Fer"), ("f3d_mind", "Ffd"), ("f3d_lncc", "FfdL"), ("f3d_mind_lncc", "FfdML"), ("mcpo", "Mc"),
                        ("mcpo_pose_fer", "Mpf"), ("convexadam", "Ca"), ("deeds", "De"), ("syn", "Sy")):
            fo, sd = col(rr, f"{key}_fold"), col(rr, f"{key}_sdlogj")
            ok = np.isfinite(sd)
            if not ok.any():
                continue
            put(f"{g}{mk}Fold", ffold(fo[ok].mean(), fo[ok].max() == 0),
                f"folding %, mean over cases, {key}, {names[g]} (n {int(ok.sum())})", float(fo[ok].mean()))
            put(f"{g}{mk}FoldMax", ffold(fo[ok].max(), fo[ok].max() == 0),
                f"folding %, maximum, {key}, {names[g]}", float(fo[ok].max()))
            put(f"{g}{mk}Sdj", f2(sd[ok].mean()), f"SDLogJ, mean over cases, {key}, {names[g]}", float(sd[ok].mean()))
            put(f"{g}{mk}SdjMax", f2(sd[ok].max()), f"SDLogJ, maximum, {key}, {names[g]}")
        fo, sd = col(rr, "mcpo_fold_05"), col(rr, "mcpo_sdlogj_05")
        ok = np.isfinite(sd)
        if ok.any():
            put(f"{g}McNatFold", f"{fo[ok].mean():.2f}", f"MCPO folding % at its native 0.5 mm grid, {names[g]}")
            put(f"{g}McNatSdj", f2(sd[ok].mean()), f"MCPO SDLogJ at its native 0.5 mm grid, {names[g]}")
    m = np.isfinite(col(allr, "mcpo_sdlogj"))
    put("McSdjRatio", f1(col(allr, "mcpo_sdlogj")[m].mean() / col(allr, "fer_sdlogj")[m].mean()), "mean MCPO SDLogJ / mean FER SDLogJ, registrations with MCPO (both regimes)")
    put("AllMcFoldN", int((col(allr, "mcpo_fold")[m] > 0).sum()), "MCPO registrations with folding (1 mm), both regimes")
    put("AllFerFoldN", int((col(allr, "fer_fold") > 0).sum()), "FER registrations with folding, both regimes")
    pp = G["Post"]
    m = np.isfinite(col(pp, "mcpo_sdlogj"))
    put("PostMcSdjRatio", f1(col(pp, "mcpo_sdlogj")[m].mean() / col(pp, "fer_sdlogj")[m].mean()), "mean MCPO SDLogJ / mean FER SDLogJ, post-resection registrations with MCPO")
    m2 = np.isfinite(col(pp, "mcpo_pose_fer_sdlogj")) & m
    put("PostMcSdjRatioMpf", f1(col(pp, "mcpo_sdlogj")[m2].mean() / col(pp, "mcpo_pose_fer_sdlogj")[m2].mean()), "mean MCPO SDLogJ / mean SDLogJ of the refinement on the MCPO pose, same registrations")
    put("PostMcFoldN", int((col(pp, "mcpo_fold")[m] > 0).sum()), "post-resection MCPO registrations with folding (1 mm)")
    put("PostFerFoldN", int((col(pp, "fer_fold") > 0).sum()), "post-resection FER registrations with folding")
    put("PostMpfFoldN", int((col(pp, "mcpo_pose_fer_fold")[np.isfinite(col(pp, "mcpo_pose_fer_fold"))] > 0).sum()), "same, refinement on the MCPO pose")
    put("AllFerDispMean", f2(col(allr, "fer_disp_mean").mean()), "mean |u| of the refinement over the FoV, mean over cases (mm), both regimes")
    put("AllFerDispP", f2(col(allr, "fer_disp_p95").mean()), "95th percentile of |u| over the FoV, mean over cases (mm), both regimes")
    put("AllFerDispMax", f2(col(allr, "fer_disp_max").max()), "largest |u| in any FoV (mm), both regimes")
    put("AllFerClip", f"{100 * col(allr, 'fer_clip_frac').mean():.1f}", "% of grid voxels at the 2 mm bound, mean over cases, both regimes")
    put("AllFerClipMax", f"{100 * col(allr, 'fer_clip_frac').max():.1f}", "same, maximum over cases")
    put("PostFerDispMean", f2(col(pp, "fer_disp_mean").mean()), "mean |u| of the refinement over the FoV, mean over post-resection cases (mm)")
    put("PostFerDispP", f2(col(pp, "fer_disp_p95").mean()), "95th percentile of |u| over the FoV, mean over post-resection cases (mm)")
    put("PostFerClip", f"{100 * col(pp, 'fer_clip_frac').mean():.1f}", "% of grid voxels at the 2 mm bound, mean over post-resection cases")
    put("ModelNodes", f"{int(np.median(col(pp, 'model_nodes'))):,}".replace(",", "\\,"), "median number of mesh nodes, post-resection registrations")
    put("ModelNodesLo", f"{int(col(pp, 'model_nodes').min()):,}".replace(",", "\\,"), "smallest mesh (post-resection)")
    put("ModelNodesHi", f"{int(col(pp, 'model_nodes').max()):,}".replace(",", "\\,"), "largest mesh (post-resection)")
    put("CorrMd", f"{int(round(np.median(col(allr, 'corr_n')) / 1000))}\\,000", "median number of block correspondences per case (rounded to thousands)")

    # ---------------------------------------------------------------- ReMIND details
    re = G["Re"]
    later = [r for r in re if r["case"] in C.REMIND_AFTER_PAIRING]
    first = [r for r in re if r["case"] not in C.REMIND_AFTER_PAIRING]
    put("RePairingN", len(first), "ReMIND patients annotated when the pairing with the affine was adopted")
    put("ReLaterN", len(later), "ReMIND patients annotated afterwards")
    put("ReLaterAffM", f2(col(later, "affine").mean()), "affine, later ReMIND patients")
    put("ReLaterFerM", f2(col(later, "fer").mean()), "FER, later ReMIND patients")
    put("ReLaterFfdM", f2(col(later, "f3d_mind").mean()), "f3d MIND, later ReMIND patients")
    put("ReLaterFervAffN", int((col(later, "fer") < col(later, "affine")).sum()), "later ReMIND patients improved by the refinement")
    lm = np.isfinite(col(later, "mcpo"))
    put("ReLaterMcM", f2(col(later, "mcpo")[lm].mean()), f"MCPO, later ReMIND patients (n {int(lm.sum())})")
    put("ReLaterMcN", int(lm.sum()), "later ReMIND patients with MCPO")
    put("ReLaterMcFerM", f2(col(later, "fer")[lm].mean()), "FER on the later ReMIND patients with MCPO")
    put("ReFerBelowBf", f"{int((col(re, 'fer') < col(re, 'bestfit')).sum())} of {len(re)}", "ReMIND patients in which affine + FER is below the in-sample best-fit rigid")
    mm = np.isfinite(col(re, "mcpo"))
    put("ReMcFerM", f2(col(re, "fer")[mm].mean()), "FER on the ReMIND patients with MCPO")
    put("ReMcAffM", f2(col(re, "affine")[mm].mean()), "affine on the ReMIND patients with MCPO")
    t2 = np.isfinite(col(re, "mcpo_t2"))
    if t2.any():
        put("ReMcTN", int(t2.sum()), "ReMIND patients with a T2 available (MCPO run with T2 as moving)")
        put("ReMcTM", f2(col(re, "mcpo_t2")[t2].mean()), "MCPO with T2 as moving image")
        put("ReMcTFerM", f2(col(re, "fer")[t2].mean()), "FER on the same patients")
        put("ReMcTCeM", f2(np.nanmean(np.where([r.get("mcpo_moving") == "ceT1" for r in re], col(re, "mcpo"), np.nan)[t2])),
            "MCPO with ceT1 on the patients that have both")
        put("ReMcTvFerP", pfmt(wil(col(re, "fer")[t2], col(re, "mcpo_t2")[t2]))[0], "Wilcoxon FER vs MCPO-T2")
        put("ReMcTvFerPr", pfmt(wil(col(re, "fer")[t2], col(re, "mcpo_t2")[t2]))[1], "same, with its relation sign")
        put("ReMcTFerN", f"{int((col(re, 'fer')[t2] < col(re, 'mcpo_t2')[t2]).sum())} of {int(t2.sum())}", "patients in which FER < MCPO-T2")

    # ---------------------------------------------------------------- MCPO on ReMIND with the T2 (same contrast as the landmarks)
    ce = np.array([r.get("mcpo_moving") == "ceT1" for r in re])
    both = t2 & ce & mm
    put("ReMcBothN", int(both.sum()), "ReMIND patients with MCPO results for both moving images (ceT1 and T2)")
    put("ReMcBothTM", f2(col(re, "mcpo_t2")[both].mean()), "MCPO with T2 on the patients that have both")
    put("ReMcBothCeM", f2(col(re, "mcpo")[both].mean()), "MCPO with ceT1 on the patients that have both")
    put("ReMcBothTBetter", f"{int((col(re, 'mcpo_t2')[both] < col(re, 'mcpo')[both]).sum())} of {int(both.sum())}", "patients in which MCPO-T2 < MCPO-ceT1")
    put("ReMcBothP", pfmt(wil(col(re, "mcpo_t2")[both], col(re, "mcpo")[both]))[0], "Wilcoxon MCPO-T2 vs MCPO-ceT1")
    put("ReMcBothPr", pfmt(wil(col(re, "mcpo_t2")[both], col(re, "mcpo")[both]))[1], "same, with its relation sign")
    best = np.where(t2, col(re, "mcpo_t2"), col(re, "mcpo"))        # T2 where the release has one, the default otherwise
    put("ReMcPrefM", f2(best[mm].mean()), "MCPO on ReMIND taking the T2 result where it exists and the ceT1 result otherwise")
    put("ReMcPrefFerN", f"{int((col(re, 'fer')[mm] < best[mm]).sum())} of {int(mm.sum())}", "patients in which FER < that MCPO result")
    put("ReMcPrefFerNn", int((col(re, 'fer')[mm] < best[mm]).sum()), "same, count only")
    put("ReMcPrefP", pfmt(wil(col(re, "fer")[mm], best[mm]))[0], "Wilcoxon FER vs that MCPO result")
    put("ReMcPrefPr", pfmt(wil(col(re, "fer")[mm], best[mm]))[1], "same, with its relation sign")
    put("ReLaterFervMcN", f"{int((col(later, 'fer')[lm] < col(later, 'mcpo')[lm]).sum())} of {int(lm.sum())}", "later ReMIND patients in which FER < MCPO")
    put("ReLaterFervMcP", pfmt(wil(col(later, "fer")[lm], col(later, "mcpo")[lm]))[0], "Wilcoxon FER vs MCPO, later ReMIND patients")
    put("ReLaterFervMcPr", pfmt(wil(col(later, "fer")[lm], col(later, "mcpo")[lm]))[1], "same, with its relation sign")
    # evidence independent of every choice: later ReMIND patients + the external case
    ex = G["Ex"]
    put("ExFailN", int((col(ex, "affine") > 5).sum()), "external patients in which the affine leaves > 5 mm")
    put("ExMcFailN", int((col(ex, "mcpo") > 5).sum()), "external patients in which MCPO leaves > 5 mm")
    ho = later + G["Ex"]
    hm = np.isfinite(col(ho, "mcpo"))
    put("HoN", len(ho), "registrations annotated after every choice was fixed: later ReMIND patients + external case")
    put("HoFervAffN", f"{int((col(ho, 'fer') < col(ho, 'affine')).sum())} of {len(ho)}", "of them, FER < affine")
    put("HoFervAffP", pfmt(wil(col(ho, "fer"), col(ho, "affine")))[0], "Wilcoxon FER vs affine on them")
    put("HoFervAffPr", pfmt(wil(col(ho, "fer"), col(ho, "affine")))[1], "same, with its relation sign")
    put("HoFervMcN", f"{int((col(ho, 'fer')[hm] < col(ho, 'mcpo')[hm]).sum())} of {int(hm.sum())}", "of them, FER < MCPO")
    put("HoFervMcP", pfmt(wil(col(ho, "fer")[hm], col(ho, "mcpo")[hm]))[0], "Wilcoxon FER vs MCPO on them")
    put("HoFervMcPr", pfmt(wil(col(ho, "fer")[hm], col(ho, "mcpo")[hm]))[1], "same, with its relation sign")
    hp = np.isfinite(col(ho, "mcpo_pose")) & np.isfinite(col(ho, "mcpo_pose_fer"))
    put("HoMpfvMpN", f"{int((col(ho, 'mcpo_pose_fer')[hp] < col(ho, 'mcpo_pose')[hp]).sum())} of {int(hp.sum())}",
        "of them, refined MCPO pose < MCPO pose (held-out registrations with a pose)")

    # ---------------------------------------------------------------- post-resection only, dependence on the affine error
    post = G["Post"]
    a_all, f_all, d_all = col(post, "affine"), col(post, "fer"), col(post, "affine") - col(post, "fer")
    rho = stats.spearmanr(a_all, d_all)
    put("GainRho", f2(rho.correlation), "Spearman correlation between the affine error and the gain of the refinement, post-resection registrations", float(rho.pvalue))
    rho_o = stats.spearmanr((a_all + f_all) / 2, d_all)                 # Oldham: change vs the mean of the two errors (no coupling)
    put("GainRhoOld", f2(rho_o.correlation), "Spearman correlation between the gain and the mean of affine and refined error (Oldham's method)", float(rho_o.pvalue))
    put("GainRhoOldP", pfmt(rho_o.pvalue)[0], "its p-value")
    mpk = np.isfinite(col(post, "mcpo_pose"))
    rho2 = stats.spearmanr(col(post, "mcpo_pose")[mpk], (col(post, "mcpo_pose") - col(post, "mcpo_pose_fer"))[mpk])
    put("GainRhoMp", f2(rho2.correlation), "Spearman correlation between the MCPO-pose error and the gain of the refinement on it", float(rho2.pvalue))
    rho2o = stats.spearmanr(((col(post, "mcpo_pose") + col(post, "mcpo_pose_fer")) / 2)[mpk], (col(post, "mcpo_pose") - col(post, "mcpo_pose_fer"))[mpk])
    put("GainRhoMpOld", f2(rho2o.correlation), "same against the mean of the two errors (Oldham)", float(rho2o.pvalue))
    lo2 = a_all < 2.0
    put("LowAffN", int(lo2.sum()), "registrations in which the affine leaves less than 2 mm")
    put("LowAffImpN", int((d_all[lo2] > 0).sum()), "of them, improved by the refinement")
    put("LowAffGain", fs(d_all[lo2].mean()), "mean gain (affine - FER) on them (mm)")
    put("HighAffN", int((~lo2).sum()), "registrations in which the affine leaves 2 mm or more")
    put("HighAffImpN", int((d_all[~lo2] > 0).sum()), "of them, improved by the refinement")
    put("HighAffGain", fs(d_all[~lo2].mean()), "mean gain on them (mm)")
    lmp = [x for x in lms if x["cohort"] != "resect_before"]
    la = np.array([float(x["affine"]) for x in lmp])
    lf = np.array([float(x["fer"]) for x in lmp])
    put("LmImpPct", f"{100 * (lf < la).mean():.0f}", "% of landmark pairs whose error falls with the refinement, post-resection sources")
    fd = col(post, "f3d_mind") - a_all
    put("FfdWorseBigN", int((fd > 1.0).sum()), "registrations that f3d MIND-SSC worsens by more than 1 mm")
    put("FfdBetterBigN", int((fd < -0.05).sum()), "registrations that f3d MIND-SSC improves by more than 0.05 mm")
    put("FfdWorseSmallN", int((fd > 0.05).sum()), "registrations that f3d MIND-SSC worsens by more than 0.05 mm")
    put("FfdFoldN", int((col(post, "f3d_mind_fold") > 0).sum()), "post-resection registrations in which f3d MIND-SSC folds")
    put("FfdSdjMd", f2(np.median(col(post, "f3d_mind_sdlogj"))), "median SDLogJ of f3d MIND-SSC, post-resection registrations")
    put("FfdLFoldLo", f"{min(col(G['Pp'], 'f3d_lncc_fold').mean(), col(G['Re'], 'f3d_lncc_fold').mean(), col(G['Pp'], 'f3d_mind_lncc_fold').mean(), col(G['Re'], 'f3d_mind_lncc_fold').mean()):.0f}", "smallest cohort-mean folding % of the LNCC variants (public post-resection, ReMIND)")
    put("FfdLFoldHi", f"{max(col(G['Pp'], 'f3d_lncc_fold').mean(), col(G['Re'], 'f3d_lncc_fold').mean(), col(G['Pp'], 'f3d_mind_lncc_fold').mean(), col(G['Re'], 'f3d_mind_lncc_fold').mean()):.0f}", "largest")
    put("AffScaleBigN", int((col(post, "aff_scale_max") > 1.2).sum()), "post-resection registrations with a scale factor above 1.2")
    put("PostAffScaleOutN", int(((col(post, "aff_scale_min") < 0.9) | (col(post, "aff_scale_max") > 1.1)).sum()),
        "post-resection registrations with an affine scale factor outside 0.90-1.10")
    # the MCPO pose: how far it is from the affine, its rotation
    mp = np.isfinite(col(post, "mcpo_pose"))
    put("PostMpRotLo", f1(np.nanmin(col(post, "mcpo_pose_rot_deg"))), "smallest rotation of the MCPO pose (deg)")
    put("PostMpRotHi", f1(np.nanmax(col(post, "mcpo_pose_rot_deg"))), "largest rotation of the MCPO pose (deg)")
    put("PostMpFitMd", f2(np.nanmedian(col(post, "mcpo_pose_fit_mm"))), "median residual of the rigid fit to MCPO's field over its inliers (mm)")
    # RESECT post: the affine failures and the rest
    rp = G["Rpo"]
    badp = col(rp, "affine") > 5
    put("RpoAffFailN", int(badp.sum()), "RESECT post-resection registrations in which the affine leaves > 5 mm")
    put("RpoAffFailLo", f1(col(rp, "affine")[badp].min()), "smallest affine error among them")
    put("RpoAffFailHi", f1(col(rp, "affine")[badp].max()), "largest")
    put("RpoAffFailRotLo", f"{col(rp, 'aff_rot_deg')[badp].min():.0f}", "smallest affine rotation among them (deg)")
    put("RpoAffFailRotHi", f"{col(rp, 'aff_rot_deg')[badp].max():.0f}", "largest")
    put("RpoAffFailMpLo", f1(col(rp, "mcpo_pose")[badp].min()), "MCPO pose error on those registrations, smallest")
    put("RpoAffFailMpHi", f1(col(rp, "mcpo_pose")[badp].max()), "largest")
    put("RpoAffFailMcLo", f1(col(rp, "mcpo")[badp].min()), "MCPO error on those registrations, smallest")
    put("RpoAffFailMcHi", f1(col(rp, "mcpo")[badp].max()), "largest")
    put("RpoOkN", int((~badp).sum()), "the other RESECT post-resection registrations")
    put("RpoOkAffM", f2(col(rp, "affine")[~badp].mean()), "affine mean on them")
    put("RpoOkFerM", f2(col(rp, "fer")[~badp].mean()), "affine + refinement mean on them")
    put("RpoOkMcM", f2(col(rp, "mcpo")[~badp].mean()), "MCPO mean on them")
    put("RpoOkMpfM", f2(col(rp, "mcpo_pose_fer")[~badp].mean()), "MCPO pose + refinement mean on them")
    put("RpoOkFervMcP", pfmt(wil(col(rp, "fer")[~badp], col(rp, "mcpo")[~badp]))[0], "Wilcoxon affine + refinement vs MCPO on them")
    put("RpoOkFervMcPr", pfmt(wil(col(rp, "fer")[~badp], col(rp, "mcpo")[~badp]))[1], "same, with its relation sign")
    put("RpoOkFerBetterN", f"{int((col(rp, 'fer')[~badp] < col(rp, 'mcpo')[~badp]).sum())} of {int((~badp).sum())}", "of them, affine + refinement < MCPO")

    # ---------------------------------------------------------------- RaPTOR on BITE (published per-patient table)
    RIVAZ = {"01": 3.8, "02": 4.1, "03": 2.9, "04": 2.5, "05": 2.5, "06": 2.5, "07": 2.7, "08": 2.5, "09": 2.3, "10": 1.6,
             "11": 4.4, "12": 1.3, "13": 1.3}
    okr = [c for c in RIVAZ if c != "12"]
    bi = {r["case"]: r for r in G["Bi"]}
    put("BiRapM", f2(np.mean(list(RIVAZ.values()))), "RaPTOR (Rivaz 2015, Table II) on BITE, mean of its 13 per-patient values")
    fr_ = np.array([bi[c]["fer"] for c in okr]); rp_ = np.array([RIVAZ[c] for c in okr]); mc_ = np.array([bi[c]["mcpo"] for c in okr])
    put("BiRapN", len(okr), "BITE patients with a self-consistent RaPTOR row (P12 excluded)")
    put("BiFervRapN", f"{int((fr_ < rp_).sum())} of {len(okr)}", "patients in which the refined affine < RaPTOR")
    put("BiFervRapPr", pfmt(wil(fr_, rp_))[1], "Wilcoxon refined affine vs RaPTOR, with its relation sign")
    put("BiMcvRapPr", pfmt(wil(mc_, rp_))[1], "Wilcoxon MCPO vs RaPTOR, with its relation sign")

    # ---------------------------------------------------------------- landmark and image facts
    seps, floors = [], {}
    for c, p in COH.items():
        fl, dropped = [], 0
        for r in G[p]:
            mr, us, _ = C.landmarks(c, r["case"])
            f, k = clean_floor(us, mr)
            fl.append(f)
            dropped += k
            if c == "remind":
                dd = np.linalg.norm(us[:, None] - us[None], axis=-1) + np.eye(len(us)) * 1e9
                seps.append(dd.min())
        floors[p] = (float(np.mean(fl)), dropped)
        put(f"{p}Floor", f2(np.mean(fl)), f"estimated annotation floor (cleaned best-fit rigid residual), {names[p]}; pairs dropped {dropped}")
    put("ReLmMinSep", f1(min(seps)), "smallest distance between two iUS landmarks of one ReMIND patient (mm)")
    import nibabel as nib
    z = [nib.load(str(C.inputs("remind", r["case"])[1])).header.get_zooms()[:3] for r in re]
    three = [max(v) <= 1.5 for v in z]
    put("ReTtwoThreeD", int(sum(three)), "ReMIND patients whose preoperative T2 has every voxel dimension <= 1.5 mm")
    put("ReTtwoTwoD", len(z) - int(sum(three)), "the others")
    put("ReTtwoMinSlice", f"{min(max(v) for v, t in zip(z, three) if not t):.0f}", "smallest slice spacing among the others (mm)")
    put("ReTtwoMaxSlice", f"{max(max(v) for v in z):.0f}", "largest T2 slice spacing among them (mm)")
    S["remind_t2_zooms"] = [tuple(float(x) for x in v) for v in z]

    # ---------------------------------------------------------------- runtime
    tot = col(pp, "basis_s") + col(pp, "deform_s")
    for k, mk in (("basis_s", "TimeBasis"), ("modal_fit_s", "TimeModal"), ("correspondences_s", "TimeCorr"), ("fe_response_s", "TimeFe"),
                  ("deform_s", "TimeDeform")):
        v = col(pp, k)
        put(f"{mk}Md", f"{np.median(v):.0f}", f"median wall time (s), {k}, post-resection registrations")
        put(f"{mk}Lo", f"{v.min():.0f}", "minimum")
        put(f"{mk}Hi", f"{v.max():.0f}", "maximum")
    put("TimeTotalMd", f"{np.median(tot):.0f}", "median wall time (s) of the refinement (model + deformable stage), post-resection")
    tmp_ = col(pp, "mcpo_pose_fer_basis_s") + col(pp, "mcpo_pose_fer_deform_s")
    tmp_ = tmp_[np.isfinite(tmp_)]
    if len(tmp_):
        put("TimeMpfMd", f"{np.median(tmp_):.0f}", "median wall time (s) of the refinement on the MCPO pose (model + deformable stage)")
    put("TimeTotalLo", f"{tot.min():.0f}", "minimum")
    put("TimeTotalHi", f"{tot.max():.0f}", "maximum")
    put("MemPeakMd", f1(np.median(col(pp, "peak_mb")) / 1024), "median peak GPU memory (GB), post-resection")
    put("MemPeakHi", f1(col(pp, "peak_mb").max() / 1024), "maximum peak GPU memory (GB), post-resection")
    v = col(G["Pp"], "mcpo_wall_s")
    v = v[np.isfinite(v)]
    if len(v):
        put("TimeMcMd", f"{np.median(v):.0f}", "median wall time (s) of the MCPO script per case (public post-resection sets, run here)")
    v = col(pp, "f3d_mind_wall_s")
    v = v[np.isfinite(v)]
    if len(v):
        put("TimeFfdMd", f"{np.median(v):.0f}", "median wall time (s) of reg_f3d MIND-SSC")
    for key, mk in (("convexadam", "Ca"), ("deeds", "De"), ("syn", "Sy")):
        v = col(pp, f"{key}_wall_s")
        v = v[np.isfinite(v)]
        if len(v):
            put(f"Time{mk}Md", f"{max(1, round(np.median(v))):.0f}", f"median wall time (s) of {key} per case (post-resection)")
    # sensitivity of the other deformables to the field-of-view treatment (fer_extra_deformables.py variants)
    import json as _json

    def _extra_mean(d):
        js = [_json.loads(f.read_text()) for f in sorted((W / "fer" / d).glob("*/*.json"))]
        vals = [j["tre"] for j in js if j["cohort"] in POST_LM]
        return (f2(float(np.mean(vals))), len(vals)) if vals else (None, 0)
    for mk, d, what in (("ExtraCaConeM", "extra_deformables/convexadam", "ConvexAdam without masks, MR zeroed outside the dilated cone"),
                        ("ExtraCaBoxM", "extra_deformables_box/convexadam", "ConvexAdam without masks, whole MR in the iUS box"),
                        ("ExtraCaMaskConeM", "extra_deformables/convexadam_mask", "ConvexAdam with masks, MR zeroed outside the cone"),
                        ("ExtraDeBoxM", "extra_deformables_box/deeds", "deeds, whole MR in the iUS box"),
                        ("ExtraSyConeM", "extra_deformables/syn", "SyN (fixed mask), MR zeroed outside the dilated cone"),
                        ("ExtraFfdCropM", "extra_deformables/f3d_crop", "f3d MIND-SSC with the MR zeroed outside the dilated cone")):
        v, n = _extra_mean(d)
        if v is not None:
            put(mk, v, f"mean TRE (mm), {what}, post-resection (n {n})")
            js = [_json.loads(f.read_text()) for f in sorted((W / "fer" / d).glob("*/*.json"))]
            js = [j for j in js if j["cohort"] in POST_LM]
            put(mk[:-1] + "Imp", f"{sum(j['tre'] < j['affine_tre'] for j in js)} of {len(js)}",
                f"registrations whose case-mean TRE the variant lowers below the affine's, {what}")
    ta = TAB / "AFFINE_VERIFY.csv"                                  # every registration re-run; else the 13-case probe
    if not (ta.exists() and len(list(csv.DictReader(open(ta)))) == len(allr)):
        ta = TAB / "AFFINE_TIMING.csv"
    if ta.exists():
        v = np.array([float(r["wall_s"]) for r in csv.DictReader(open(ta)) if r.get("cohort", "") != "resect_before"])
        put("TimeAffMd", f"{np.median(v):.0f}", f"median wall time (s) of reg_aladin (challenge configuration), {len(v)} post-resection cases")
        put("TimeAffLo", f"{v.min():.0f}", "minimum")
        put("TimeAffHi", f"{v.max():.0f}", "maximum")
        put("TimeAffCases", len(v), "cases timed")
    fr = TAB / "AFFINE_FIRST_RUN.csv"                               # written by scripts/fer_fix_affines.py
    rr = list(csv.DictReader(open(fr))) if fr.exists() else []
    put("AffReproN", len(allr) - len(rr), "registrations whose affine was reproduced exactly by a second run of reg_aladin")
    put("AffNotReproN", len(rr), "registrations whose first affine was not reproduced (the reproducible transform is used)")
    put("AffNotReproShift", f1(max([float(r["max_landmark_shift_mm"]) for r in rr] + [0.0])),
        "largest landmark displacement between the first and the reproducible transform (mm)")
    put("AffNotReproTre", f2(max([abs(float(r["tre_used"]) - float(r["tre_first_run"])) for r in rr] + [0.0])),
        "largest change of the case-mean affine TRE between the two (mm)")

    # ---------------------------------------------------------------- second run of the stage (scripts/fer_steps.py)
    st = TAB / "STEPS.csv"
    if st.exists():
        sr = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(st))}
        if len(sr) == len(allr):
            d2 = np.array([abs(float(sr[(r["cohort"], r["case"])]["both"]) - float(r["fer"])) for r in allr])
            put("RepeatMaxDiff", f2(d2.max()), "largest |case-mean TRE of a second run - reported run| of the refinement (mm)")
            put("RepeatMeanDiff", f2(d2.mean()), "mean of the same (mm)")
            cm = max(abs(np.mean([float(sr[(r["cohort"], r["case"])]["both"]) for r in G[g]]) - col(G[g], "fer").mean())
                     for g in ("Rpo", "Bi", "Rpr", "Re"))
            put("RepeatCohortMaxDiff", f2(cm), "largest change of a cohort mean between the two runs (mm)")
            put("ModalDispMean", f2(np.mean([float(sr[(r["cohort"], r["case"])]["modal_disp_mean"]) for r in pp])),
                "mean |u_m| of the modal fit over the FoV, mean over post-resection cases (mm)")
            # the two steps apart (supplement): modal fit alone, local response alone, both
            for g in ("Rpo", "Bi", "Re", "Ex", "Post", "Rpr"):
                for key, mk in (("modal", "StepModal"), ("response", "StepResp"), ("both", "StepBoth")):
                    v = np.array([float(sr[(r["cohort"], r["case"])][key]) for r in G[g]])
                    put(f"{g}{mk}M", f2(v.mean()), f"mean TRE of the refinement with {key} only (both = complete, second run), {names[g]}")
                v = np.array([float(sr[(r["cohort"], r["case"])]["modal_disp_mean"]) for r in G[g]])
                put(f"{g}StepModalDisp", f2(v.mean()), f"mean |u_m| of the modal fit alone over the FoV (mm), {names[g]}")
            for key, mk in (("modal", "StepModal"), ("response", "StepResp")):
                a = np.array([float(sr[(r["cohort"], r["case"])][key]) for r in pp])
                b = np.array([float(sr[(r["cohort"], r["case"])]["both"]) for r in pp])
                af = col(pp, "affine")
                put(f"Post{mk}vAffN", f"{int((a < af).sum())} of {len(a)}", f"post-resection cases in which {key} alone < affine")
                put(f"Post{mk}vAffP", pfmt(wil(a, af))[0], f"Wilcoxon {key} alone vs affine, post-resection")
                put(f"Post{mk}vAffPr", pfmt(wil(a, af))[1], "same, with its relation sign")
                put(f"PostBothv{mk}N", f"{int((b < a).sum())} of {len(a)}", f"post-resection cases in which both steps < {key} alone")
                put(f"PostBothv{mk}P", pfmt(wil(b, a))[0], f"Wilcoxon both steps vs {key} alone, post-resection")
                put(f"PostBothv{mk}Pr", pfmt(wil(b, a))[1], "same, with its relation sign")
                put(f"PostBothv{mk}D", f2(abs(float((a - b).mean()))), f"|mean ({key} alone - both)| over post-resection cases (mm)")

    # ---------------------------------------------------------------- tables
    def best(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return min(v) if v else None

    hdr = {"Rpo": ("RESECT", "post-res."), "Bi": ("BITE", "post-res."), "Rpr": ("RESECT", "pre-res."), "Re": ("ReMIND", "post-res."),
           "Ex": ("External", "site")}
    cols = ["Rpo", "Bi", "Re", "Ex"]                                       # the article: post-resection only
    mov = {"Rpo": "FLAIR", "Bi": "ceT1", "Rpr": "FLAIR", "Re": "T2", "Ex": "FLAIR (one T1)"}
    cite = {"Rpo": r"RESECT post-resection \citep{Xiao2017}", "Bi": r"BITE group 4 \citep{Mercier2012}",
            "Rpr": r"RESECT pre-resection \citep{Xiao2017}", "Re": r"ReMIND \citep{Juvekar2024}",
            "Ex": "External site"}
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Evaluation set & Cases & Pairs & Moving MR & Initial (mm) & Best-fit rigid (mm) & Floor (mm)\\", r"\midrule"]
    for pcol in cols:
        L.append(f"{cite[pcol]} & {MAC[pcol + 'N']} & {MAC[pcol + 'Pairs']} & {mov[pcol]} & {MAC[pcol + 'IniM']} & {MAC[pcol + 'BfM']} & "
                 + MAC[f"{pcol}Floor"] + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_cohorts.tex").write_text("\n".join(L) + "\n")

    def cell(g, arm, bold=False):
        k = f"{g}{ARM[arm]}M"
        if k not in MAC:
            return "---"
        txt = MAC[k] + (f" $\\pm$ {MAC[g + ARM[arm] + 'S']}" if f"{g}{ARM[arm]}S" in MAC else "")
        if g == "Re" and arm in ("mcpo", "mcpo_pose", "mcpo_pose_fer"):
            txt += "$^{a}$"
        return f"\\textbf{{{txt}}}" if bold else txt
    L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(hdr[pcol][0] for pcol in cols) + r"\\",
         " & " + " & ".join(f"{hdr[pcol][1]} ({MAC[pcol + 'N']})" for pcol in cols) + r"\\", r"\midrule"]
    lab = [("initial", "Initial (navigation)"), ("affine", "NiftyReg affine (challenge baseline)"),
           ("f3d_mind", r"\quad + free-form stage (f3d, MIND-SSC)"), ("convexadam", r"\quad + ConvexAdam"),
           ("deeds", r"\quad + deeds"), ("syn", r"\quad + SyN"), ("fer", r"\quad + FE refinement (proposed)"),
           ("mcpo_pose", "MCPO pose (rigid part of MCPO)"), ("mcpo_pose_fer", r"\quad + FE refinement (proposed)"),
           ("mcpo", "MCPO, complete (authors' code)")]
    methods = ["affine", "f3d_mind", "convexadam", "deeds", "syn", "fer", "mcpo_pose", "mcpo_pose_fer", "mcpo"]
    for arm, name in lab:
        cells = []
        for pcol in cols:
            vals = {m: (float(MAC[f"{pcol}{ARM[m]}M"]) if f"{pcol}{ARM[m]}M" in MAC else None) for m in methods}
            bm = best([vals[m] for m in methods])
            is_best = arm in methods and vals[arm] is not None and abs(vals[arm] - bm) < 0.005
            cells.append(cell(pcol, arm, is_best))
        if arm == "mcpo_pose":
            L.append(r"\midrule")
        L.append(f"{name} & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    L.append("Improved by the refinement, from the affine & " + " & ".join(f"{MAC[pcol + 'FervAffN']}/{MAC[pcol + 'N']}" for pcol in cols) + r"\\")
    L.append(r"\quad $p$ vs.\ affine (Holm) & " + " & ".join(MAC.get(pcol + "FervAffPh", "---") for pcol in cols) + r"\\")
    L.append("Improved by the refinement, from the MCPO pose & " + " & ".join(
        (f"{MAC[pcol + 'MpfvMpN']}/{MAC[pcol + 'MpN']}" if f"{pcol}MpfvMpN" in MAC else "---") for pcol in cols) + r"\\")
    L.append(r"\quad $p$ vs.\ MCPO pose (Holm) & " + " & ".join(MAC.get(pcol + "MpfvMpPh", "---") for pcol in cols) + r"\\")
    L.append(r"\midrule")
    L.append("Best-fit rigid (bound) & " + " & ".join(MAC[pcol + "BfM"] for pcol in cols) + r"\\")
    L.append("Estimated annotation floor & " + " & ".join(MAC[pcol + 'Floor'] for pcol in cols) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_main.tex").write_text("\n".join(L) + "\n")

    def reg(g, mk):
        if f"{g}{mk}Sdj" not in MAC:
            return "---", "---"
        return (f"{MAC[g + mk + 'Fold']} ({MAC[g + mk + 'FoldMax']})", f"{MAC[g + mk + 'Sdj']} ({MAC[g + mk + 'SdjMax']})")
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{RESECT post + BITE (" + MAC["PpN"] + r")} & \multicolumn{2}{c}{ReMIND (" + MAC["ReN"] + r")} & "
         r"\multicolumn{2}{c}{External (" + MAC["ExN"] + r")} & Time\\",
         r" & Folding (\%) & SDLogJ & Folding (\%) & SDLogJ & Folding (\%) & SDLogJ & median (s)\\", r"\midrule"]
    for mk, name, tk in (("Ffd", "Affine + f3d, MIND-SSC", "TimeFfdMd"), ("FfdL", "Affine + f3d, LNCC", None),
                         ("FfdML", "Affine + f3d, MIND-SSC + LNCC", None), ("Ca", "Affine + ConvexAdam", "TimeCaMd"),
                         ("De", "Affine + deeds", "TimeDeMd"), ("Sy", "Affine + SyN", "TimeSyMd"),
                         ("Fer", "Affine + FE refinement", "TimeTotalMd"),
                         ("Mpf", "MCPO pose + FE refinement", "TimeMpfMd"), ("Mc", "MCPO, complete", "TimeMcMd")):
        ra, rb = reg("Pp", mk)
        rc, rd = reg("Re", mk)
        re_, rf = reg("Ex", mk)
        L.append(f"{name} & {ra} & {rb} & {rc} & {rd} & {re_} & {rf} & {MAC.get(tk, '---') if tk else '---'}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_fields.tex").write_text("\n".join(L) + "\n")

    # supplementary: per-case table (both regimes)
    short = {"resect_after": "RESECT post", "bite_g4": "BITE", "resect_before": "RESECT pre", "remind": "ReMIND", "external": "External"}

    def fcell(r, k):
        return f"{r[k]:.2f}" if np.isfinite(r.get(k, np.nan)) else "---"
    head = r"Set & Case & Pairs & Initial & Affine & + f3d & + FER & MCPO pose & + FER & MCPO & Best-fit\\"
    L = [r"\begin{longtable}{llrrrrrrrrr}", r"\caption{Case-mean TRE (mm) of every registration. Affine: NiftyReg challenge baseline; + f3d: its free-form stage "
         r"with the MIND-SSC descriptor (Section S3); + FER: the FE refinement on the preceding initialization; MCPO pose: rigid transform fitted to MCPO's field; "
         r"MCPO: the complete method, authors' code from the navigation alignment (ReMIND: contrast-enhanced T1 as moving image; "
         r"ReMIND-009, absent from the challenge release, on a 0.5 mm grid over the iUS volume, Section S4); best-fit: six-degree-of-freedom rigid "
         r"transform fitted to the landmarks. Text figures are computed from unrounded values; differences of rounded entries can deviate by 0.01 mm. The MCPO pose was not "
         r"computed for the pre-resection regime. $^{*}$ReMIND patients annotated after the pairing with the affine was adopted (Section 4.2 of the article).}\label{tab:percase}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
         r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for c in ("resect_after", "bite_g4", "remind", "external", "resect_before"):
        for r in G[COH[c]]:
            cs = r["case"].replace("ReMIND-", "").replace("Case", "").replace("Subject-", "S")
            if c == "remind" and r["case"] in C.REMIND_AFTER_PAIRING:
                cs += r"$^{*}$"                                  # annotated after the pairing with the affine was adopted
            L.append(f"{short[c]} & {cs} & {int(r['n'])} & {r['initial']:.2f} & {r['affine']:.2f} & {fcell(r, 'f3d_mind')} & {r['fer']:.2f} & "
                     f"{fcell(r, 'mcpo_pose')} & {fcell(r, 'mcpo_pose_fer')} & {fcell(r, 'mcpo')} & {r['bestfit']:.2f}" + r"\\")
        L.append(r"\midrule")
    L[-1] = r"\end{longtable}"
    (TEX / "supp_percase.tex").write_text("\n".join(L) + "\n")

    # supplementary: confidence intervals and success rates (post-resection cohorts)
    arms_ci = (("initial", "Initial"), ("affine", "Affine"), ("f3d_mind", "+ f3d MIND-SSC"), ("fer", "+ FE refinement"),
               ("mcpo_pose", "MCPO pose"), ("mcpo_pose_fer", "+ FE refinement"), ("mcpo", "MCPO, complete"))
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Set & Arm & Mean [95\% CI] & Median & P70 & $\le$3\,mm (\%) & $>$5\,mm\\", r"\midrule"]
    for pcol in ("Rpo", "Bi", "Re", "Ex"):
        for arm, name in arms_ci:
            v = col(G[pcol], arm)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            lo, hi = boot_ci(v)
            ak = ARM[arm]
            L.append(f"{names[pcol] if arm == 'initial' else ''} & {name}{' ($n=' + str(len(v)) + '$)' if len(v) != len(G[pcol]) else ''} & "
                     f"{v.mean():.2f} [{lo:.2f}, {hi:.2f}] & {np.median(v):.2f} & {MAC.get(pcol + ak + 'T', '---')} & "
                     f"{MAC[pcol + ak + 'Ok']} & {MAC[pcol + ak + 'Bad']}" + r"\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (TEX / "supp_ci.tex").write_text("\n".join(L) + "\n")

    # supplementary: the pre-resection regime of RESECT (not part of the article's problem)
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Arm & Mean $\pm$ SD & Median & P70 & Cases improved vs.\ affine ($p$)\\", r"\midrule"]
    for arm, name in (("initial", "Initial (navigation)"), ("affine", "NiftyReg affine"), ("f3d_mind", "\\quad + f3d, MIND-SSC"),
                      ("fer", "\\quad + FE refinement"), ("mcpo", "MCPO, complete"), ("bestfit", "Best-fit rigid (bound)")):
        ak = ARM[arm]
        imp = ("---" if arm in ("initial", "affine", "bestfit") else f"{MAC['Rpr' + ak + 'vAffNofT']} ({MAC['Rpr' + ak + 'vAffP']})")
        L.append(f"{name} & {MAC['Rpr' + ak + 'M']} $\\pm$ {MAC['Rpr' + ak + 'S']} & {MAC['Rpr' + ak + 'Md']} & {MAC.get('Rpr' + ak + 'T', '---')} & {imp}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "supp_pre.tex").write_text("\n".join(L) + "\n")

    # supplementary: deformable stages on the challenge affine (f3d variants and the other methods)
    L = [r"\begin{tabular}{lccccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{RESECT post + BITE (" + MAC["PpN"] + r")} & \multicolumn{3}{c}{ReMIND (" + MAC["ReN"] + r")} & "
         r"\multicolumn{3}{c}{External (" + MAC["ExN"] + r")}\\",
         r"Stage on the challenge affine & TRE & Fold.\ (\%) & SDLogJ & TRE & Fold.\ (\%) & SDLogJ & TRE & Fold.\ (\%) & SDLogJ\\",
         r"\midrule", f"None (affine alone) & {MAC['PpAffM']} & 0 & 0 & {MAC['ReAffM']} & 0 & 0 & {MAC['ExAffM']} & 0 & 0" + r"\\"]
    for mk, name in (("Ffd", "f3d, MIND-SSC"), ("FfdL", "f3d, LNCC"), ("FfdML", "f3d, MIND-SSC + LNCC"), ("Ca", "ConvexAdam"),
                     ("De", "deeds"), ("Sy", "SyN"), ("Fer", "FE refinement (proposed)")):
        L.append(f"{name} & " + " & ".join(f"{MAC.get(g + mk + 'M', '---')} & {MAC.get(g + mk + 'Fold', '---')} & {MAC.get(g + mk + 'Sdj', '---')}"
                                           for g in ("Pp", "Re", "Ex")) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "supp_f3d.tex").write_text("\n".join(L) + "\n")

    # supplementary: the two steps apart (second run of the stage on the challenge affine)
    if "PostStepBothM" in MAC:
        L = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Set & Affine & Modal fit alone & Local FE response alone & Both (the refinement) & $|u_{\mathrm{m}}|$ (mm)\\", r"\midrule"]
        for g, name in (("Rpo", "RESECT post-resection"), ("Bi", "BITE"), ("Re", "ReMIND"), ("Ex", "External site"),
                        ("Post", "All post-resection"), ("Rpr", "RESECT pre-resection")):
            L.append(f"{name} & {MAC[g + 'AffM']} & {MAC[g + 'StepModalM']} & {MAC[g + 'StepRespM']} & {MAC[g + 'StepBothM']} & {MAC[g + 'StepModalDisp']}" + r"\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (TEX / "supp_steps.tex").write_text("\n".join(L) + "\n")

    # supplementary: sensitivity on BITE (development cohort)
    sp = TAB / "SENSITIVITY.csv"
    if sp.exists():
        sr = list(csv.DictReader(open(sp)))
        ref = {r["case"]: float(r["tre"]) for r in sr if r["variant"] == "frozen"}
        lab = [("frozen", r"Frozen ($\lambda_{\mathrm{eig}}{=}3\times10^{-3}$, $\lambda_{\mathrm{rel}}{=}10$, bound 2\,mm)"),
               ("lam_eig_1e-3", r"$\lambda_{\mathrm{eig}}=10^{-3}$"), ("lam_eig_1e-2", r"$\lambda_{\mathrm{eig}}=10^{-2}$"),
               ("lam_rel_3", r"$\lambda_{\mathrm{rel}}=3$"), ("lam_rel_30", r"$\lambda_{\mathrm{rel}}=30$"),
               ("bound_1mm", r"bound 1\,mm"), ("bound_3mm", r"bound 3\,mm")]
        L = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Configuration & TRE & Mean change & Largest $|$change$|$ & SDLogJ & Max.\ folding (\%)\\", r"\midrule"]
        sens = {}
        for key, name in lab:
            rr = [r for r in sr if r["variant"] == key]
            v = np.array([float(r["tre"]) for r in rr])
            dv = v - np.array([ref[r["case"]] for r in rr])
            sens[key] = dict(tre=v.mean(), diff=dv.mean(), maxabs=np.abs(dv).max())
            L.append(f"{name} & {v.mean():.2f} & " + ("---" if key == "frozen" else f"{dv.mean():+.2f}".replace("-", "$-$").replace("+", "$+$"))
                     + " & " + ("---" if key == "frozen" else f"{np.abs(dv).max():.2f}")
                     + f" & {np.mean([float(r['sdlogj']) for r in rr]):.2f} & {max(float(r['fold_pct']) for r in rr):.2f}" + r"\\")
            if key == "frozen":
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}"]
        (TEX / "supp_sens.tex").write_text("\n".join(L) + "\n")
        S["sensitivity"] = sens
        dd = [abs(v["diff"]) for k, v in sens.items() if k != "frozen"]
        put("SensMaxMeanChange", f2(max(dd)), "largest |mean change| of BITE TRE over the six one-at-a-time variants of the constants")
        put("SensMaxCaseChange", f2(max(v["maxabs"] for k, v in sens.items() if k != "frozen")), "largest per-case |change| over the variants")
        put("SensTreLo", f2(min(v["tre"] for v in sens.values())), "lowest BITE mean TRE over the variants")
        put("SensTreHi", f2(max(v["tre"] for v in sens.values())), "highest BITE mean TRE over the variants")

    # ---------------------------------------------------------------- write
    out = ["% Generated by paper/tools/build_numbers.py -- do not edit by hand.", "% Source: fer/tables/PER_CASE.csv, LANDMARKS.csv"]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MAC.items()]
    (TEX / "numbers.tex").write_text("\n".join(out) + "\n")
    md = ["# NUMBERS.md: every macro of numbers.tex", "",
          "Generated by `paper/tools/build_numbers.py` from `fer/tables/PER_CASE.csv` and `LANDMARKS.csv` "
          "(written by `scripts/fer_score.py` from the registration outputs and the landmark files).", "",
          "| macro | value | definition |", "|---|---|---|"]
    md += [f"| `\\{k}` | {v} | {d} |" for k, v, d, _ in DOC]
    (TOOLS / "NUMBERS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "stats.json").write_text(json.dumps(S, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    # every macro used by the manuscript must exist
    import re
    used = set()
    for f in ("main.tex", "supplementary.tex"):
        if (TEX / f).exists():
            used |= set(re.findall(r"\\((?:Rpo|Bi|Rpr|Re|Ex|Pub|Rs|All|Pat|Aff|Time|Mem|Mc|Model|Holm|Corr|Sens|Post|Gain|Low|High|Lm|Ffd|Ho|Repeat|Modal|Pp|Rpo|Step|Bi)[A-Za-z]+)", (TEX / f).read_text(encoding="utf-8")))
    known_tex = {"Re", "Rs", "Ex", "All", "Time", "Bi", "Mc"}
    missing = sorted(u for u in used if u not in MAC and u not in known_tex)
    print(f"{len(MAC)} macros written; used in the manuscript: {len(used)}; undefined: {missing}")
    for k in ("PubAffM", "PubFerM", "PubFervAffNofT", "ReAffM", "ReFerM", "ReFervAffNofT", "ExAffM", "ExFerM", "ReMcM", "ReMcFerM",
              "RpoFerM", "BiFerM", "RprFerM", "RpoMcM", "BiMcM", "RprMcM", "AllFervAffNofT", "AllFfdvAffNofT"):
        print(f"  {k} = {MAC.get(k)}")


if __name__ == "__main__":
    main()
