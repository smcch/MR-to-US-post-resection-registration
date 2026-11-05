"""Global pose search over an A-PRIORI PHYSICALLY CONSTRAINED box.

The published search used +-20 deg with a 10 deg step and then discarded
implausible winners with a fitted gate. That is the wrong order: on BITE folder
11 the metric's global optimum is a 52.2 deg pose worth 34.54 mm, while a
correct rigid exists at 1.5 deg worth 5.44 mm. Landmark-derived pose rotations
never exceed 7.5 deg on BITE or 8.8 deg on RESECT -- a patient-to-image
registration error of 52 deg is not a registration error, it is an optimisation
failure.

So constrain the space instead of filtering the answer: +-10 deg with a 5 deg
step. The rotation count is unchanged (5^3 = 125), so the cost is the same, the
angular resolution doubles, and the catastrophic basin is excluded by
construction rather than by a threshold fitted on the cohort being reported.

Usage: python rg_search_tight.py --cohort bite_g4 [--case 01 02 ...]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigid_search_fft as RS                                        # noqa
from rg_cohorts import COHORTS, OUT, cohort_cases                    # noqa

TIGHT = dict(coarse_spacing=2.0, fine_spacing=1.0,
             t_max=25.0, t_step=5.0,
             r_max=10.0, r_step=5.0,           # <-- the only change
             topk=12, sep=8.0, batch=48,      # K24 over-searched (3.42 mm, sd 2.04); K12 gives 3.25, sd 1.38
             levels=(4, 2, 1), iters=(200, 150, 120),
             max_bg=0.02, rot_cap=10.0, trans_cap=30.0,
             mind_mode="recompute", mmean_mode="crop",
             refine_mode="full", tier_k=6)


def install_clamped_descend(cap_deg):
    """Project the rotation vector onto the +-cap_deg ball after every Adam step.

    Constraining only the coarse grid is not enough: the refinement is a free
    6-DoF Adam descent and walks straight back out of the box (measured on BITE
    folder 11, coarse box +-10 deg, winner returned at 52.4 deg). The constraint
    has to live inside the optimiser.
    """
    import numpy as np
    import torch
    import capture_range as CR

    # capture_range.rodrigues() takes the rotation vector IN DEGREES, so the
    # cap is in degrees too. Converting it to radians silently clamped every
    # pose to 0.17 deg and made the whole search translation-only.
    cap = float(cap_deg)
    orig = CR.CaseCost.descend

    def descend(self, r0, t0, levels=(4, 2, 1), iters=(200, 150, 120),
                lr_t=1.0, lr_r=0.4, decay=0.1, cache=None):
        cur_r = np.asarray(r0, np.float32).copy()
        cur_t = np.asarray(t0, np.float32).copy()
        n0 = float(np.linalg.norm(cur_r))
        if n0 > cap:                       # start inside the feasible set
            cur_r *= cap / n0
        cache = {} if cache is None else cache
        for lev, nit in zip(levels, iters):
            if lev not in cache:
                cache[lev] = self.pyramid(lev)
            mf, k, X, ks = cache[lev]
            r = torch.tensor(cur_r, device=self.dev, requires_grad=True)
            t = torch.tensor(cur_t, device=self.dev, requires_grad=True)
            s = float(lev) ** 0.5
            opt = torch.optim.Adam([{"params": [t], "lr": lr_t * s},
                                    {"params": [r], "lr": lr_r * s}])
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda i, n=nit: 1.0 - (1.0 - decay) * i / max(n - 1, 1))
            best = (float(self.cost_torch(r, t, mf, k, X, ks)),
                    cur_r.copy(), cur_t.copy())
            for _ in range(nit):
                opt.zero_grad()
                loss = self.cost_torch(r, t, mf, k, X, ks)
                loss.backward()
                opt.step()
                sch.step()
                with torch.no_grad():      # <-- projection onto the ball
                    nr = float(r.norm())
                    if nr > cap:
                        r.mul_(cap / nr)
                lv = float(loss)
                if lv < best[0]:
                    best = (lv, r.detach().cpu().numpy().copy(),
                            t.detach().cpu().numpy().copy())
            cur_r, cur_t = best[1], best[2]
        rr, tt = cur_r.astype(float), cur_t.astype(float)
        return rr, tt, float(self.cost_poses(np.concatenate([rr, tt])[None])[0])

    CR.CaseCost.descend = descend
    print(f"[tight] rotation clamped to +-{cap_deg} deg inside the refinement")
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import rg_cohorts
    rg_cohorts.install(a.cohort)          # rebind the accessors inside RS
    install_clamped_descend(TIGHT["rot_cap"])
    root = Path(a.out) if a.out else Path(OUT) / "search_tight" / a.cohort
    root.mkdir(parents=True, exist_ok=True)
    cases = a.case or cohort_cases(a.cohort)
    rows = []
    for c in cases:
        try:
            r = RS.search(c, a.cohort, dict(TIGHT), out_root=root)
            rows.append(r)
            print(f"[tight] {c}: TRE {r.get('tre'):.3f}  rot {r.get('rot_deg'):.1f}  "
                  f"|t| {r.get('trans_mm'):.1f}  oracle {r.get('tre_oracle'):.3f}",
                  flush=True)
        except Exception as e:                                        # noqa: BLE001
            print(f"[tight] {c}: ERROR {e}", flush=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(root / "PER_CASE.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", root / "PER_CASE.csv")


if __name__ == "__main__":
    main()
