"""
analysis.py
-----------
Two things the headline table cannot tell you.

1. PAIRED COMPARISON. Seed-to-seed spread in this setup is about +/- 0.015 AUC, because
   every seed regenerates the graph and the cascade. The model differences we care about
   are around +0.005. Comparing marginal means is therefore useless - the error bars
   swallow the effect. But the models share a seed, so the comparison can be paired:
   for each seed, take (model - baseline) and ask whether that difference is consistent.
   Pairing removes the shared data variance, which is the entire problem.

2. FRACTION OF HEADROOM RECOVERED. The diagnostic oracle (logistic regression handed the
   exact generative feature) bounds what any model could gain from structure. Expressing
   the GNN's gain as a share of that bound says something an absolute AUC cannot:
   not "is it better" but "how much of what was there did it actually get".

Run:  python src/analysis.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

BASE = "Logistic Regression (handcrafted only)"
PROP = "Full model: GAT + handcrafted  (proposed)"
ORACLE = "[diagnostic] LR + true structural diversity"


def paired(per_seed: dict, metric: str = "auc_roc"):
    def vals(name):
        return np.array([r[metric] for r in per_seed[name]], dtype=float)

    base, prop, orc = vals(BASE), vals(PROP), vals(ORACLE)
    n = min(len(base), len(prop), len(orc))
    base, prop, orc = base[:n], prop[:n], orc[:n]

    d = prop - base                      # what the GNN actually gained
    h = orc - base                       # headroom the oracle says was available
    out = {
        "n_seeds": int(n),
        "baseline": base.tolist(),
        "proposed": prop.tolist(),
        "oracle": orc.tolist(),
        "delta_per_seed": d.tolist(),
        "delta_mean": float(d.mean()),
        "delta_sd": float(d.std(ddof=1)) if n > 1 else 0.0,
        "headroom_mean": float(h.mean()),
        "share_of_headroom_recovered": float(d.mean() / h.mean()) if h.mean() else None,
        "wins": int((d > 0).sum()),
    }
    if n > 1 and d.std(ddof=1) > 0:
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        out["paired_t"] = float(t)
        # Three seeds is far too few for a p-value to mean much. The t statistic is
        # reported as a consistency signal, not as evidence of significance, and the
        # honest read is the sign pattern across seeds.
    return out


def main():
    with open(os.path.join(RESULTS, "metrics.json")) as f:
        m = json.load(f)

    report = {}
    for label, key in [("regime_A_default", "per_seed"),
                       ("regime_B_structural", "per_seed_structural")]:
        ps = m.get(key)
        if not ps or BASE not in ps:
            continue
        report[label] = {met: paired(ps, met) for met in ("auc_roc", "auc_pr")}

    with open(os.path.join(RESULTS, "paired_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Paired analysis: does the GNN actually beat the baseline?", "",
             "Every model in a given row shares its seed with the baseline it is "
             "compared against, so the differences below are paired. Marginal means "
             "with +/- 0.015 error bars would hide an effect of +0.005; pairing does "
             "not.", ""]
    for label, block in report.items():
        lines += [f"## {label.replace('_', ' ')}", ""]
        lines += ["| Metric | baseline | proposed | per-seed delta | mean delta | "
                  "seeds won | oracle headroom | share recovered |",
                  "|---|---|---|---|---|---|---|---|"]
        for met, r in block.items():
            deltas = ", ".join(f"{x:+.4f}" for x in r["delta_per_seed"])
            share = ("n/a" if r["share_of_headroom_recovered"] is None
                     else f"{100 * r['share_of_headroom_recovered']:.0f}%")
            lines.append(
                f"| {met} | {np.mean(r['baseline']):.4f} | {np.mean(r['proposed']):.4f} "
                f"| {deltas} | {r['delta_mean']:+.4f} ± {r['delta_sd']:.4f} "
                f"| {r['wins']}/{r['n_seeds']} | {r['headroom_mean']:+.4f} | {share} |")
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "paired_analysis.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
