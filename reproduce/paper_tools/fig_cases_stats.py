"""Fig. 3 of the FE-refinement manuscript, from fer/tables/PER_CASE.csv (post-resection registrations only).
Author's choice (2026-10-03): the figures show only the baseline, the proposed method and MCPO.
(a) per-case change of the error produced by the FE refinement applied on the challenge affine (refinement - affine),
    against the error of the affine.
(b) regularity: SDLogJ over the iUS field of view of the proposed method and of MCPO, one point per registration
    (fields at 1 mm).
Vector PDF + 300 dpi PNG; Okabe-Ito palette.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

W = Path("/mnt/c/ReMIND_work")
OUT = W / "paper/MIA/imagenes"
MM = 1 / 25.4
INK, MUTED = "#222222", "#6b6b6b"
COH = [("resect_after", "RESECT post-resection", "s", "#D55E00"), ("bite_g4", "BITE", "^", "#009E73"),
       ("remind", "ReMIND", "D", "#CC79A7"), ("external", "External site", "*", "#222222")]
POST = [c[0] for c in COH]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


rows = [{k: (v if k in ("cohort", "case", "mcpo_moving") else num(v)) for k, v in r.items()}
        for r in csv.DictReader(open(W / "fer/tables/PER_CASE.csv"))]
rows = [r for r in rows if r["cohort"] in POST]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(170 * MM, 92 * MM), gridspec_kw=dict(width_ratios=[1.45, 1.0]))

# ------------------------------------------------------------------ (a)
YLO, YHI = -1.65, 0.5      # -1.65: the largest decrease (ReMIND-024, -1.51 mm) was clipped at -1.5
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
ax.axhspan(0, YHI, color="#D55E00", alpha=0.05, lw=0, zorder=0)
for key, lab, mk, colr in COH:
    rr = [r for r in rows if r["cohort"] == key]
    ax.plot([r["affine"] for r in rr], [r["fer"] - r["affine"] for r in rr], mk, color=colr, ms=5.5 if mk != "*" else 9.5,
            mew=0.7, mec="white", ls="", zorder=3, label=lab)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 20])
ax.set_xticklabels(["1", "2", "3", "5", "10", "20"])
ax.set_xlim(0.9, 24)
ax.set_ylim(YLO, YHI)
ax.set_xlabel("TRE after the NiftyReg affine (mm)", fontsize=8.5, color=INK)
ax.set_ylabel("Change produced by the FE refinement (mm)", fontsize=8.5, color=INK)
ax.text(0.985, 0.035, "improves the affine", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
ax.text(0.985, 0.965, "worsens the affine", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
ax.set_title("(a)", fontsize=9, loc="left", color=INK)
ax.legend(fontsize=7, frameon=False, loc="lower left", handletextpad=0.3, labelspacing=0.35)

# ------------------------------------------------------------------ (b)
METH = [("fer_sdlogj", "Affine +\nFE refinement", "#0072B2"), ("mcpo_sdlogj", "MCPO", "#E69F00")]
FLOOR = 0.008
rng = np.random.default_rng(0)
for i, (k, lab, colr) in enumerate(METH):
    v = np.array([r.get(k, np.nan) for r in rows], float)
    v = v[np.isfinite(v)]
    vv = np.maximum(v, FLOOR)
    bx.plot(i + rng.uniform(-0.22, 0.22, len(vv)), vv, "o", color=colr, ms=3.2, mew=0, alpha=0.75, zorder=3)
    bx.plot([i - 0.32, i + 0.32], [max(np.median(v), FLOOR)] * 2, color=INK, lw=1.6, zorder=4)
    bx.text(i, 2.3, f"median {np.median(v):.2f}", ha="center", va="center", fontsize=7, color=INK)
bx.set_yscale("log")
bx.set_ylim(0.03, 3)
bx.set_yticks([0.03, 0.1, 0.3, 1])
bx.set_yticklabels(["0.03", "0.1", "0.3", "1"])
bx.set_xticks(range(len(METH)))
bx.set_xticklabels([m[1] for m in METH], fontsize=7.5)
bx.set_ylabel("SDLogJ over the iUS field of view", fontsize=8.5, color=INK)
bx.set_xlim(-0.6, len(METH) - 0.4)
bx.set_title("(b)", fontsize=9, loc="left", color=INK)
for a in (ax, bx):
    a.tick_params(labelsize=7.5, colors=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
fig.tight_layout(pad=0.5, w_pad=1.5)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig3_cases.pdf")
fig.savefig(OUT / "fig3_cases.png", dpi=300)
n_fer = sum(1 for r in rows if r["fer"] < r["affine"])
worst = max(r["fer"] - r["affine"] for r in rows)
print(f"fig2 ok: {len(rows)} post-resection registrations; FER below its affine in {n_fer}; largest increase {worst:.2f} mm")
