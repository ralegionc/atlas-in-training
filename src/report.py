"""Every number the README quotes, printed from results/ (run analysis.py first).

  python report.py > results/report.md
"""
import json
import os

import numpy as np
import pandas as pd

import analysis as A
import config as cfg

R = cfg.RESULTS


def read(name, size=None):
    path = os.path.join(R, "{}_{}.csv".format(name, size) if size else name + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def tokens(step):
    return step * cfg.TOKENS_PER_STEP / 1e9


def main():
    s = json.load(open(os.path.join(R, "summary.json")))
    print("# Report\n")
    print("places in the analysis:", s.get("places"))

    sc = read("scaling")
    print("\n## Scaling: median error, km\n")
    print(sc.pivot(index="region", columns="model", values="median_km")
          .reindex(columns=["spelling"] + cfg.SIZES).round(-1).to_string())
    rep = read("replication")
    if rep is not None:
        print("\n## Replication of Godey et al. (2024), R^2 x 100\n")
        print(rep.to_string(index=False))

    tc = read("template_check")
    if tc is not None:
        print("\n## Template check, {} final: median km and cut beyond spelling\n".format(cfg.TRACK_MODEL))
        print(tc.pivot(index="region", columns="prompt", values="median_km").round(0).to_string())
        print(tc.pivot(index="region", columns="prompt", values="vs_spell").round(3).to_string())
        a = tc.pivot(index="region", columns="prompt", values="vs_spell").drop("World")
        print("rank correlation of regional cuts between templates: {:.2f}".format(
            a["name"].rank().corr(a["bare"].rank())))

    for size in [cfg.TRACK_MODEL, cfg.BIG_MODEL]:
        cur = read("region_curves", size)
        if cur is None:
            continue
        print("\n## {}: formation\n".format(size))
        w = cur[cur.region == "World"].set_index("step").median_km
        print("world median km by step:", {int(k): round(v) for k, v in w.items()})
        arr = read("arrival", size)
        print("\nhalf of each region's improvement reached at step (95% CI):")
        for r in arr.itertuples():
            print("  {:30s} {:>8,.0f}  [{:,.0f}, {:,.0f}]  {:,.0f} -> {:,.0f} km".format(
                r.region, r.half_step, r.half_lo, r.half_hi, cur[(cur.region == r.region)].median_km.iloc[0],
                r.final_km))
        bs = read("beyond_spelling", size)
        wb = bs[bs.region == "World"].set_index("step")
        print("\nworld gain beyond spelling by step:", {int(k): round(100 * v, 1) for k, v in wb.gain.items()})
        last = bs[bs.step == bs.step.max()].sort_values("gain", ascending=False)
        print("\nfinal gain beyond spelling by region (95% CI):")
        for r in last.itertuples():
            print("  {:30s} {:+5.1f}%  [{:+.1f}, {:+.1f}]  model {:,.0f} km, spelling {:,.0f} km".format(
                r.region, 100 * r.gain, 100 * r.lo, 100 * r.hi, r.model_km, r.spell_km))
        ss = s.get(size, {})
        for by in ["region", "iso"]:
            st = ss.get("letters_first_" + by)
            if st:
                print("\nletters first by {} (n {}): early rho {:.2f} (p {:.1g}), partial {:.2f}; "
                      "late rho {:.2f} (p {:.2g}), partial {:.2f} (p {:.2g})".format(
                          by, st["n"], st["rho_early"], st["p_early"], st["partial_early"],
                          st["rho_late"], st["p_late"], st["partial_late"], st["partial_p_late"]))
        cg = read("country_gain", size)
        if cg is not None:
            st = ss["country_gain_rho"]
            print("\ncountry gain vs country mentions: rho {:.2f} (p {:.2g}, n {})".format(
                st["rho"], st["p"], st["n"]))
            print("lowest:", ", ".join("{} {:+.0%}".format(r.country, r.gain)
                                       for r in cg.nsmallest(6, "gain").itertuples()))
            print("highest:", ", ".join("{} {:+.0%}".format(r.country, r.gain)
                                        for r in cg.nlargest(6, "gain").itertuples()))

    for size in [cfg.TRACK_MODEL, cfg.BIG_MODEL, "2.8b"]:
        cm = read("count_models", size)
        if cm is None:
            continue
        cm = cm.set_index(cm.columns[0])
        print("\n## {}: counts at the final checkpoint (log10 km per tenfold, cluster-robust)\n".format(size))
        for k, r in cm.iterrows():
            print("  {:28s} {:+.4f}  [{:+.4f}, {:+.4f}]  p {:.2g}  n {:,.0f}".format(k, r.beta, r.lo, r.hi, r.p, r.n))
        ss = s.get(size, {})
        if "country_slope" in ss:
            c = ss["country_slope"]
            print("  country level: log km on log mentions {:+.3f} [{:+.3f}, {:+.3f}], r {:.2f}".format(
                c["beta"], c["lo"], c["hi"], ss["country_r"]))
        gp = read("gap", size)
        if gp is not None:
            print("\n  region gaps (log10 km vs NW Europe): raw -> +spelling -> +counts")
            for r in gp.itertuples():
                print("  {:30s} {:+.3f} -> {:+.3f} -> {:+.3f}   by spelling {:+.0%}, by counts {:+.0%}, counts only {:+.0%}"
                      .format(r.region, r.raw, r.spelling, r.counts, r.by_spelling, r.by_counts, r.by_counts_only))
        if "alpha" in ss:
            print("\n  exposure alpha:", {k: (round(v["alpha"], 2), round(v["lo"], 2), round(v["hi"], 2))
                                         for k, v in ss["alpha"].items()})
            print("  country arrival vs mentions:", ss.get("country_arrival"))


if __name__ == "__main__":
    main()
