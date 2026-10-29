"""Every number in the README comes from here.

  1. formation   error by region at every checkpoint of the tracked model, and the step at
                 which each region reaches half its final skill ("arrival")
  2. counts      at the final checkpoint, does a place's error track how often the Pile
                 names it, within its own country? Does naming it next to its country add
                 anything, and does naming it next to a placebo country not?
  3. gap         how much of each region's extra error do the counts account for?
  4. exposure    over training: is a place's error a function of how many mentions the
                 model has seen so far (count x step / 143,000), whatever the step? The
                 exponent alpha in alpha*log(count) + log(step) says: alpha 1 means
                 exposure is all that matters, alpha 0 means counts do not matter at all
  5. scaling     final checkpoints from 70m to 2.8b, the spelling floor, the replication

Skill is 1 - median error / median null error, where the null guesses the training mean
for every place, so 0 is no knowledge and 1 a perfect map. Places are in the 15,767
sample of places.py less the 4.5% with a distant namesake (homonyms.py); intervals are 95% bootstrap over places, or cluster-robust over
countries for regressions.

  python analysis.py            # writes results/*.csv and results/summary.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config as cfg   # not "C": patsy formulas use C() for categoricals

R = cfg.RESULTS
REF_REGION = "Northern & Western Europe"


def places(strict=False):
    """The sample, minus names that GeoNames also gives to a town 300+ km away (the
    model may know the other one). strict: also minus every name Wikipedia had to
    disambiguate ("Gainesville, Georgia")."""
    pl = pd.read_parquet(os.path.join(cfg.DATA, "places.parquet"))
    hom = pd.read_parquet(os.path.join(cfg.DATA, "homonyms.parquet"))
    pl = pl.merge(hom, on="pid", how="left")
    keep = pl.n_far.eq(0)
    if strict:
        keep &= ~pl.had_suffix
    return pl[keep].reset_index(drop=True)


def preds(tag):
    path = os.path.join(cfg.PREDS, tag + ".parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def tracked(size=cfg.TRACK_MODEL, steps=cfg.TRACK_STEPS):
    rows = []
    for s in steps:
        p = preds(cfg.tag(size, s))
        if p is not None:
            rows.append(p.assign(step=s))
    return pd.concat(rows, ignore_index=True)


def counts():
    path = os.path.join(cfg.DATA, "counts.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def skill(err, null):
    return 1 - np.median(err) / np.median(null)


def boot_skill(err, null, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), (n, len(err)))
    s = 1 - np.median(err[idx], 1) / np.median(null[idx], 1)
    return np.percentile(s, [2.5, 97.5])


# 1. formation ------------------------------------------------------------------------

def region_curves(long, pl):
    d = long.merge(pl[["pid", "region"]], on="pid")
    rows = []
    for (s, reg), g in pd.concat([d, d.assign(region="World")]).groupby(["step", "region"]):
        rows.append(dict(step=s, region=reg, n=len(g), median_km=g.err_km.median(),
                         null_km=g.null_km.median(), skill=skill(g.err_km, g.null_km)))
    return pd.DataFrame(rows)


def log_interp_step(steps, values, target):
    """First step (log-interpolated) at which values reach target; steps > 0."""
    for i in range(1, len(steps)):
        if values[i] >= target > values[i - 1] or (i == 1 and values[0] >= target):
            if values[i - 1] >= target:
                return steps[i - 1]
            f = (target - values[i - 1]) / (values[i] - values[i - 1])
            return float(np.exp(np.log(steps[i - 1]) + f * (np.log(steps[i]) - np.log(steps[i - 1]))))
    return np.nan


def arrival(curves, frac=0.5):
    """Step at which a region's skill first covers frac of the way from its step-0 skill
    to its final skill. Step 0 sits at step 1 on the log axis."""
    out = []
    for reg, g in curves.groupby("region"):
        g = g.sort_values("step")
        st = np.maximum(g.step.values, 1).astype(float)
        sk = g.skill.values
        lo, hi = sk[0], sk[-1]
        out.append(dict(region=reg, start_skill=lo, final_skill=hi,
                        final_km=g.median_km.values[-1],
                        half_step=log_interp_step(st, sk, lo + frac * (hi - lo))))
    return pd.DataFrame(out).sort_values("half_step")


def arrival_ci(long, pl, n=200, seed=0):
    """Bootstrap the half-skill step by resampling places within each region."""
    rng = np.random.default_rng(seed)
    d = long.merge(pl[["pid", "region"]], on="pid")
    steps = sorted(d.step.unique())
    res = {}
    for reg, g in d.groupby("region"):
        pids = g.pid.unique()
        E = g.pivot(index="pid", columns="step", values="err_km").loc[pids, steps].values
        N = g.pivot(index="pid", columns="step", values="null_km").loc[pids, steps].values
        st = np.maximum(np.array(steps), 1).astype(float)
        vals = []
        for _ in range(n):
            idx = rng.integers(0, len(pids), len(pids))
            sk = 1 - np.median(E[idx], 0) / np.median(N[idx], 0)
            vals.append(log_interp_step(st, sk, sk[0] + 0.5 * (sk[-1] - sk[0])))
        res[reg] = np.nanpercentile(vals, [2.5, 97.5])
    return res


def beyond_spelling(long, pl):
    """What training adds that the letters do not say. At each checkpoint the probe's
    and the spelling baseline's out-of-fold predictions are pooled by a linear map (six
    inputs, three outputs) fit on the other folds and applied to the held-out one. The
    gain is how much lower the pool's median error is than the same pool built on the
    untrained network, which is the spelling baseline plus a probe on random features:
    so every region starts at exactly zero, and re-weighting the spelling baseline
    (which by itself helps regions near the middle of the sample) is not counted."""
    import geo
    sp = preds("spell").set_index("pid").loc[pl.pid]
    S = geo.to_xyz(sp.pred_lat.values, sp.pred_lon.values)
    Y = geo.to_xyz(pl.lat.values, pl.lon.values)
    folds = pl.fold.values
    e_sp = geo.gc_km(S, Y)
    groups = [("World", np.arange(len(pl)))] + list(pl.groupby("region").indices.items())
    pooled = {}
    for s, g in long.groupby("step"):
        g = g.set_index("pid").loc[pl.pid]
        M = geo.to_xyz(g.pred_lat.values, g.pred_lon.values)
        X = np.hstack([M, S, np.ones((len(M), 1))])
        P = np.zeros_like(Y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            W, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
            P[te] = X[te] @ W
        pooled[s] = (geo.gc_km(P, Y), g.err_km.values)
    e0 = pooled[min(pooled)][0]
    rows = []
    for s, (e, em) in sorted(pooled.items()):
        for reg, idx in groups:
            rows.append(dict(step=s, region=reg, pooled_km=np.median(e[idx]),
                             pooled0_km=np.median(e0[idx]), spell_km=np.median(e_sp[idx]),
                             model_km=np.median(em[idx]),
                             gain=1 - np.median(e[idx]) / np.median(e0[idx]),
                             vs_spell=1 - np.median(e[idx]) / np.median(e_sp[idx])))
    return pd.DataFrame(rows)


def beyond_ci(long, pl, step=cfg.FINAL_STEP, n=1000, seed=0):
    """Bootstrap interval of the final gain per region, resampling places."""
    import geo
    rng = np.random.default_rng(seed)
    out = {}
    sp = preds("spell").set_index("pid").loc[pl.pid]
    S = geo.to_xyz(sp.pred_lat.values, sp.pred_lon.values)
    Y = geo.to_xyz(pl.lat.values, pl.lon.values)
    folds = pl.fold.values
    errs = []
    for s in (0, step):
        g = long[long.step == s].set_index("pid").loc[pl.pid]
        X = np.hstack([geo.to_xyz(g.pred_lat.values, g.pred_lon.values), S, np.ones((len(g), 1))])
        P = np.zeros_like(Y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            W, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
            P[te] = X[te] @ W
        errs.append(geo.gc_km(P, Y))
    groups = [("World", np.arange(len(pl)))] + list(pl.groupby("region").indices.items())
    for reg, idx in groups:
        b = rng.choice(idx, (n, len(idx)))
        g = 1 - np.median(errs[1][b], 1) / np.median(errs[0][b], 1)
        out[reg] = np.percentile(g, [2.5, 97.5])
    return out


def country_gain(long, pl, cc, min_places=30, step=cfg.FINAL_STEP):
    """Per country: the final gain beyond spelling (as in beyond_spelling), with how
    often the Pile names the country."""
    bs_rows = []
    sub = pl.copy()
    sub["region"] = sub.iso      # reuse beyond_spelling with countries as the groups
    bs = beyond_spelling(long[long.step.isin([0, step])], sub)
    bs = bs[(bs.step == step) & (bs.region != "World")].rename(columns={"region": "iso"})
    n = pl.groupby("iso").size().rename("n")
    out = bs.merge(n, left_on="iso", right_index=True)
    out = out[out.n >= min_places].merge(cc[["iso", "country_count"]], on="iso", how="left")
    out = out.merge(pl.groupby("iso")[["country", "region"]].first(), left_on="iso", right_index=True)
    out["log_country_count"] = np.log10(out.country_count)
    return out


def letters_first(long, pl, by="region", split=2000, min_places=30):
    """Does early progress go where spelling is informative? Per group: how far below the
    untrained network's error the spelling baseline gets (what the letters can say), and
    the share of the untrained error the model removes before and after `split`."""
    from scipy import stats
    sp = preds("spell").merge(pl[["pid", by]], on="pid")
    d = long.merge(pl[["pid", by]], on="pid")
    rows = []
    for key, g in d.groupby(by):
        if g.pid.nunique() < min_places:
            continue
        m = g.groupby("step").err_km.median()
        e0, e1, e2 = m.loc[0], m.loc[split], m.loc[m.index.max()]
        rows.append(dict(group=key, n=g.pid.nunique(), e0=e0,
                         spell_cut=1 - sp[sp[by] == key].err_km.median() / e0,
                         early_cut=(e0 - e1) / e0, late_cut=(e1 - e2) / e0))
    out = pd.DataFrame(rows)

    def partial(x, y, z):
        """Rank correlation of x and y with the rank of z partialled out: both cuts
        divide by the starting error, which must not drive the link."""
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        Z = np.column_stack([stats.rankdata(z), np.ones(len(x))])
        ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        return stats.pearsonr(ex, ey)

    res = {"n": len(out)}
    for k in ["early", "late"]:
        r = stats.spearmanr(out.spell_cut, out[k + "_cut"])
        q = partial(out.spell_cut, out[k + "_cut"], np.log(out.e0))
        res.update({"rho_" + k: float(r.statistic), "p_" + k: float(r.pvalue),
                    "partial_" + k: float(q.statistic), "partial_p_" + k: float(q.pvalue)})
    return out, res


# 2. counts ---------------------------------------------------------------------------

def place_table(final, pl, ct):
    d = final.merge(pl, on="pid").merge(ct[["pid", "count", "own", "placebo", "ntok_llama",
                                            "count_approx", "own_approx", "country_count"]
                                           + (["placebo_approx"] if "placebo_approx" in ct else [])],
                                        on="pid", how="left")
    sp = preds("spell")
    if sp is not None:
        d = d.merge(sp[["pid", "err_km"]].rename(columns={"err_km": "spell_km"}), on="pid", how="left")
        d["log_spell"] = np.log10(d.spell_km.clip(lower=1))
    for c in ["count", "own", "placebo", "ntok_llama"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")   # columns not yet fetched arrive as None
    d["log_err"] = np.log10(d.err_km.clip(lower=1))
    d["log_null"] = np.log10(d.null_km.clip(lower=1))
    d["log_count"] = np.log10(d["count"] + 1)
    d["log_own"] = np.log10(d["own"] + 1)
    d["log_placebo"] = np.log10(d["placebo"] + 1)
    d["log_views"] = np.log10(d.page_views)
    d["log_country"] = np.log10(pd.to_numeric(d.country_count, errors="coerce"))
    d["etype"] = d.entity_type
    return d


def ols(formula, d, cluster="iso"):
    d = d.dropna(subset=[c for c in d.columns if c in formula])
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return m


def resid(formula, d):
    """OLS residuals, evaluated here so that C() in the formula is patsy's."""
    return smf.ols(formula, d).fit().resid


def coef(m, name):
    lo, hi = m.conf_int().loc[name]
    return dict(beta=m.params[name], lo=lo, hi=hi, p=m.pvalues[name], n=int(m.nobs))


def count_models(d):
    """Within-country slopes of log error on log counts; country fixed effects absorb
    everything about where a country is, how big it is and how much it is written about."""
    base = "log_err ~ log_count + C(iso) + C(etype) + log_views + log_null + ntok_llama"
    out = {}
    m = ols(base, d)
    out["count"] = coef(m, "log_count")
    m = ols(base.replace("log_count", "log_count + log_own"), d)
    out["count|own"], out["own|count"] = coef(m, "log_count"), coef(m, "log_own")
    have = d.dropna(subset=["placebo"])
    if len(have) > 500:
        m = ols(base.replace("log_count", "log_count + log_own + log_placebo"), have)
        out["own|placebo"] = coef(m, "log_own")
        out["placebo|own"] = coef(m, "log_placebo")
        m = ols(base.replace("log_count", "log_count + log_placebo"), have)
        out["placebo alone"] = coef(m, "log_placebo")
    m = ols(base.replace("log_count", "log_views"), d.assign(log_views=d.log_views))
    out["views only"] = coef(m, "log_views")
    # names that spell like somewhere else: does the count still matter?
    m = ols(base + " + log_spell", d)
    out["count|spell"] = coef(m, "log_count")
    # names that are also ordinary words: their count is mostly the word
    m = ols(base + " + C(wordlike)", d)
    out["count|wordlike"] = coef(m, "log_count")
    m = ols(base, d[~d.wordlike.astype(bool)])
    out["count, not word-like"] = coef(m, "log_count")
    m = ols(base.replace("log_count", "log_count + log_own"), d[~d.wordlike.astype(bool)])
    out["own|count, not word-like"] = coef(m, "log_own")
    # rarer names only: no subsampling anywhere
    exact = d[(d.own_approx == False) & (d.count_approx == False)]  # noqa: E712
    m = ols(base.replace("log_count", "log_count + log_own"), exact)
    out["own|count exact"] = coef(m, "log_own")
    # the spelling baseline's own error held fixed: what the letters make easy
    if "log_spell" in d:
        m = ols(base.replace("log_count", "log_count + log_own + log_spell"), d)
        out["own|count,spell"] = coef(m, "log_own")
        out["count|own,spell"] = coef(m, "log_count")
    # what training changed: log error at initialisation minus at the end
    if "log_init" in d:
        learned = base.replace("log_err ~", "learned ~")
        m = ols(learned.replace("log_count", "log_count + log_own + log_spell"),
                d.assign(learned=d.log_init - d.log_err))
        out["learned: own|count,spell"] = coef(m, "log_own")
        out["learned: count|own,spell"] = coef(m, "log_count")
    return pd.DataFrame(out).T


def country_level(d, cc, min_places=20):
    g = d.groupby("iso").agg(n=("pid", "size"), median_km=("err_km", "median"),
                             null_km=("null_km", "median"), region=("region", "first"),
                             country=("country", "first"), mean_log_count=("log_count", "mean"))
    g = g[g.n >= min_places].reset_index().merge(cc, on="iso", how="left")
    g["skill"] = 1 - g.median_km / g.null_km
    g["log_country_count"] = np.log10(g.country_count)
    g["log_km"] = np.log10(g.median_km)
    return g


# 3. gap ------------------------------------------------------------------------------

def gap(d):
    """Region coefficients: log10 error relative to Northern & Western Europe.

      raw       geometry only (distance from the middle of the sample, place type)
      spelling  + how far the spelling baseline is off (names that read like elsewhere)
      counts    + Pile mentions of the place, mentions next to its country, mentions of
                its country, page views, name length
      counts_only  geometry + the corpus terms, no spelling term

    The share of a region's gap each step removes is (raw - next) / raw."""
    d = d.copy()
    d["region"] = pd.Categorical(d.region, [REF_REGION] + sorted(set(d.region) - {REF_REGION}))
    geo_ = "log_err ~ C(region) + log_null + C(etype)"
    corpus = " + log_count + log_own + log_country + log_views + ntok_llama"
    fits = {"raw": ols(geo_, d), "spelling": ols(geo_ + " + log_spell", d),
            "counts": ols(geo_ + " + log_spell" + corpus, d),
            "counts_only": ols(geo_ + corpus, d)}
    rows = []
    for reg in d.region.cat.categories[1:]:
        k = "C(region)[T.{}]".format(reg)
        r = dict(region=reg)
        for name, m in fits.items():
            lo, hi = m.conf_int().loc[k]
            r.update({name: m.params[k], name + "_lo": lo, name + "_hi": hi})
        rows.append(r)
    out = pd.DataFrame(rows)
    out["by_spelling"] = (out.raw - out.spelling) / out.raw
    out["by_counts"] = (out.spelling - out.counts) / out.raw
    out["by_counts_only"] = (out.raw - out.counts_only) / out.raw
    out.attrs["slopes"] = {n: {t: float(m.params[t]) for t in ["log_spell", "log_count", "log_own"]
                               if t in m.params} for n, m in fits.items()}
    return out


# 4. exposure -------------------------------------------------------------------------

def binned_sse(z, y, bins=40):
    q = np.quantile(z, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, z, side="right") - 1, 0, bins - 1)
    mu = np.bincount(b, y, bins) / np.maximum(np.bincount(b, minlength=bins), 1)
    return np.sum((y - mu[b]) ** 2)


def exposure_table(long, ct, col="count", min_step=512):
    """One row per place and checkpoint with log error, log10(col + 1) and log10 step."""
    d = long[long.step >= min_step].merge(ct[["pid", "iso", col]], on="pid").dropna(subset=[col])
    d["y"] = np.log10(d.err_km.clip(lower=1).values)
    d["lc"] = np.log10(d[col].values + 1)
    d["ls"] = np.log10(d.step.values)
    return d


def exposure_alpha(d, grid=np.linspace(0, 2, 81)):
    """Fit log error = f(alpha log10(count+1) + log10 step) with f a step function on 40
    quantile bins; alpha minimises the squared error."""
    sse = np.array([binned_sse(a * d.lc.values + d.ls.values, d.y.values) for a in grid])
    return grid[np.argmin(sse)], grid, sse


def exposure_alpha_ci(d, unit="pid", n=100, seed=0, grid=np.linspace(0, 2, 81)):
    """Bootstrap over units (places, or countries when the count is a country's)."""
    rng = np.random.default_rng(seed)
    by = d.groupby(unit).indices
    keys = list(by)
    y, lc, ls = d.y.values, d.lc.values, d.ls.values
    best = []
    for _ in range(n):
        picks = rng.choice(len(keys), len(keys))
        idx = np.concatenate([by[keys[k]] for k in picks])
        sse = [binned_sse(a * lc[idx] + ls[idx], y[idx]) for a in grid]
        best.append(grid[int(np.argmin(sse))])
    return np.percentile(best, [2.5, 97.5])


def country_arrival(long, pl, cc, min_places=30):
    """Half-progress step of each country with enough places, and how often the Pile names it."""
    d = long.merge(pl[["pid", "iso", "country", "region"]], on="pid")
    rows = []
    for iso, g in d.groupby("iso"):
        if g.pid.nunique() < min_places:
            continue
        m = g.groupby("step").err_km.median().sort_index()
        st = np.maximum(m.index.values, 1).astype(float)
        km = m.values
        prog = (km[0] - km) / (km[0] - km[-1]) if km[0] > km[-1] else np.full(len(km), np.nan)
        rows.append(dict(iso=iso, country=g.country.iloc[0], region=g.region.iloc[0],
                         n=g.pid.nunique(), start_km=km[0], final_km=km[-1],
                         half_step=log_interp_step(st, prog, 0.5) if km[0] > km[-1] else np.nan))
    out = pd.DataFrame(rows).merge(cc[["iso", "country_count"]], on="iso", how="left")
    out["log_country_count"] = np.log10(out.country_count)
    return out


def step_slopes(long, pl, ct):
    """At each checkpoint, the within-country slope of log error on log count."""
    rows = []
    for s, g in long.groupby("step"):
        d = place_table(g.drop(columns="step"), pl, ct)
        m = ols("log_err ~ log_count + C(iso) + C(etype) + log_views + log_null + ntok_llama", d)
        rows.append(dict(step=s, **coef(m, "log_count")))
    return pd.DataFrame(rows)


def count_bins(long, ct, q=4):
    """Places never counted form bin 0; the rest are split into q quantile bins."""
    d = long.merge(ct[["pid", "count"]], on="pid").dropna(subset=["count"])
    pos = d["count"] > 0
    d["bin"] = 0
    d.loc[pos, "bin"] = 1 + pd.qcut(np.log10(d.loc[pos, "count"]), q, labels=False, duplicates="drop")
    edges = d.groupby("bin")["count"].agg(["min", "max", "median"])
    g = d.groupby(["bin", "step"]).agg(median_km=("err_km", "median"), null_km=("null_km", "median"),
                                       median_count=("count", "median")).reset_index()
    g["skill"] = 1 - g.median_km / g.null_km
    g["exposures"] = g.median_count * g.step / cfg.FINAL_STEP
    return g, edges


def template_check(pl, size=cfg.TRACK_MODEL):
    """The final checkpoint read as "Where is X" and as the bare name: median error and
    the pooled cut beyond spelling, by region."""
    import geo
    sp = preds("spell").set_index("pid").loc[pl.pid]
    S = geo.to_xyz(sp.pred_lat.values, sp.pred_lon.values)
    Y = geo.to_xyz(pl.lat.values, pl.lon.values)
    folds = pl.fold.values
    rows = []
    for prompt in ["name", "bare"]:
        p = preds(cfg.tag(size, cfg.FINAL_STEP, prompt))
        if p is None:
            return None
        p = p.set_index("pid").loc[pl.pid]
        X = np.hstack([geo.to_xyz(p.pred_lat.values, p.pred_lon.values), S, np.ones((len(p), 1))])
        P = np.zeros_like(Y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            W, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
            P[te] = X[te] @ W
        e = geo.gc_km(P, Y)
        es = geo.gc_km(S, Y)
        for reg, idx in [("World", np.arange(len(pl)))] + list(pl.groupby("region").indices.items()):
            rows.append(dict(prompt=prompt, region=reg, median_km=np.median(p.err_km.values[idx]),
                             vs_spell=1 - np.median(e[idx]) / np.median(es[idx])))
    return pd.DataFrame(rows)


# 5. scaling --------------------------------------------------------------------------

def scaling(pl):
    rows = []
    for size in cfg.SIZES:
        p = preds(cfg.tag(size, cfg.FINAL_STEP))
        if p is None:
            continue
        d = p.merge(pl[["pid", "region"]], on="pid")
        rows.append(dict(model=size, region="World", median_km=d.err_km.median(),
                         skill=skill(d.err_km, d.null_km)))
        for reg, g in d.groupby("region"):
            rows.append(dict(model=size, region=reg, median_km=g.err_km.median(),
                             skill=skill(g.err_km, g.null_km)))
    sp = preds("spell")
    if sp is not None:
        d = sp.merge(pl[["pid", "region"]], on="pid")
        rows.append(dict(model="spelling", region="World", median_km=d.err_km.median(),
                         skill=skill(d.err_km, d.null_km)))
        for reg, g in d.groupby("region"):
            rows.append(dict(model="spelling", region=reg, median_km=g.err_km.median(),
                             skill=skill(g.err_km, g.null_km)))
    return pd.DataFrame(rows)


def replication():
    rows = []
    for size in cfg.SIZES:
        path = os.path.join(cfg.PREDS, cfg.tag(size, cfg.FINAL_STEP, "title") + ".replicate.csv")
        if os.path.exists(path):
            r = pd.read_csv(path)
            rows.append(dict(model=size, r2=100 * r.r2.max(), layer=int(r.layer[r.r2.idxmax()])))
    return pd.DataFrame(rows)


def formation(size, steps, pl, summary):
    """Sections 1 and 1b for one tracked model. Files carry the model size."""
    long = tracked(size, steps)
    long = long[long.pid.isin(pl.pid)]
    if long.step.nunique() < 3:
        print(size, "not enough checkpoints yet")
        return None
    f = lambda name: os.path.join(R, "{}_{}.csv".format(name, size))
    curves = region_curves(long, pl)
    curves.to_csv(f("region_curves"), index=False)
    arr = arrival(curves)
    ci = arrival_ci(long, pl)
    arr["half_lo"] = arr.region.map(lambda r: ci.get(r, [np.nan, np.nan])[0])
    arr["half_hi"] = arr.region.map(lambda r: ci.get(r, [np.nan, np.nan])[1])
    arr.to_csv(f("arrival"), index=False)
    print(arr.to_string(index=False))
    bs = beyond_spelling(long, pl)
    if bs.step.max() == cfg.FINAL_STEP:
        bci = beyond_ci(long, pl)
        last = bs.step == cfg.FINAL_STEP
        bs["lo"] = np.where(last, bs.region.map(lambda r: bci[r][0]), np.nan)
        bs["hi"] = np.where(last, bs.region.map(lambda r: bci[r][1]), np.nan)
    bs.to_csv(f("beyond_spelling"), index=False)
    out = {"steps": sorted(int(s) for s in long.step.unique())}
    for by in ["region", "iso"]:
        lf, st = letters_first(long, pl, by)
        lf.to_csv(f("letters_first_" + by), index=False)
        out["letters_first_" + by] = st
        print(size, "letters first by", by, st)
    cc = pd.read_parquet(os.path.join(cfg.DATA, "country_counts.parquet"))
    if cfg.FINAL_STEP in set(long.step):
        from scipy import stats
        cg = country_gain(long, pl, cc)
        cg.to_csv(f("country_gain"), index=False)
        r = stats.spearmanr(cg.log_country_count, cg.gain)
        out["country_gain_rho"] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(cg))
    summary[size] = out
    return long


def counts_section(size, long, pl, ct, cc, summary):
    """Sections 2 to 4 for one model: counts at the final checkpoint, the gap, exposure."""
    f = lambda name: os.path.join(R, "{}_{}.csv".format(name, size))
    out = summary.setdefault(size, {})
    final = long[long.step == cfg.FINAL_STEP].drop(columns="step")
    d = place_table(final, pl, ct)
    init = preds(cfg.tag(size, 0))
    if init is not None:
        d = d.merge(init[["pid", "err_km"]].rename(columns={"err_km": "init_km"}), on="pid", how="left")
        d["log_init"] = np.log10(d.init_km.clip(lower=1))
    cm = count_models(d)
    cm.to_csv(f("count_models"))
    print(size, "count models")
    print(cm.to_string())
    cl = country_level(d, cc)
    cl.to_csv(f("countries"), index=False)
    m = ols("log_km ~ log_country_count", cl)
    out["country_slope"] = coef(m, "log_country_count")
    out["country_r"] = float(np.corrcoef(cl.log_country_count, cl.log_km)[0, 1])
    gp = gap(d)
    gp.to_csv(f("gap"), index=False)
    out["gap_slopes"] = gp.attrs["slopes"]
    print(gp.round(3).to_string(index=False))
    if long.step.nunique() < 5:
        return
    alphas = {}
    for col, unit in [("count", "pid"), ("own", "pid"), ("country_count", "iso")]:
        dd = exposure_table(long, ct, col)
        a, grid, sse = exposure_alpha(dd)
        lo, hi = exposure_alpha_ci(dd, unit)
        alphas[col] = dict(alpha=float(a), lo=float(lo), hi=float(hi), n=int(dd[unit].nunique()))
        pd.DataFrame(dict(alpha=grid, sse=sse)).to_csv(f("alpha_" + col), index=False)
        print(size, "alpha", col, alphas[col])
    out["alpha"] = alphas
    ca = country_arrival(long, pl, cc)
    ca.to_csv(f("country_arrival"), index=False)
    ok = ca.dropna(subset=["half_step", "log_country_count"])
    from scipy import stats
    r = stats.spearmanr(np.log10(ok.half_step), ok.log_country_count)
    out["country_arrival"] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(ok))
    step_slopes(long, pl, ct).to_csv(f("step_slopes"), index=False)
    for col in ["count", "own"]:
        cb, edges = count_bins(long, ct.assign(count=ct[col]))
        cb.to_csv(f("count_bins_" + col), index=False)
        edges.to_csv(f("count_bin_edges_" + col))


def main():
    os.makedirs(R, exist_ok=True)
    pl = places()
    path = os.path.join(R, "summary.json")
    summary = {}
    longs = {}
    for size, steps in [(cfg.TRACK_MODEL, cfg.TRACK_STEPS), (cfg.BIG_MODEL, cfg.BIG_STEPS)]:
        longs[size] = formation(size, steps, pl, summary)

    sc = scaling(pl)
    sc.to_csv(os.path.join(R, "scaling.csv"), index=False)
    print(sc[sc.region == "World"].to_string(index=False))
    rep = replication()
    if len(rep):
        rep.to_csv(os.path.join(R, "replication.csv"), index=False)
        print(rep.to_string(index=False))
    summary["places"] = len(pl)
    tc = template_check(pl)
    if tc is not None:
        tc.to_csv(os.path.join(R, "template_check.csv"), index=False)
        print(tc.pivot(index="region", columns="prompt", values="median_km").round(0).to_string())

    ct = counts()
    ready = ct is not None and ct["count"].notna().mean() > 0.95 and "own" in ct
    if ready:
        cc = pd.read_parquet(os.path.join(cfg.DATA, "country_counts.parquet"))
        summary["own_done"] = float(ct.own.notna().mean())
        summary["placebo_done"] = int(ct.placebo.notna().sum()) if "placebo" in ct else 0
        for size in [cfg.TRACK_MODEL, cfg.BIG_MODEL, "2.8b"]:
            long = longs.get(size)
            if long is None:
                p = preds(cfg.tag(size, cfg.FINAL_STEP))
                if p is None:
                    continue
                long = p.assign(step=cfg.FINAL_STEP)
                long = long[long.pid.isin(pl.pid)]
            if cfg.FINAL_STEP in set(long.step):
                counts_section(size, long, pl, ct, cc, summary)
    else:
        print("counts not ready")
    json.dump(summary, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
