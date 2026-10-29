"""Linear probes from hidden states to where a place is.

The target is the place's position as a unit vector (x, y, z), so the probe has no seam
at the date line and no stretching near the poles. A prediction is turned back into a
direction and scored by great-circle distance in km.

For each kept layer: standardise, ridge regression, the penalty chosen by efficient
leave-one-out on the training folds. The layer is chosen the same way, inside each
training fold, by leave-one-out error, so no test place informs any choice. Every place
gets one out-of-fold prediction.

The ridge is FastRidgeCV below: the same estimator and the same leave-one-out
criterion as sklearn's RidgeCV (tests/test_core.py checks they agree), computed from
one eigendecomposition, which is several times faster for wide hidden states.

Two floors come with it:
  null   the training folds' mean direction for every place (no information at all)
  spell  character 1-4 grams of the name -> ridge (what spelling alone gives: "-pur" is
         South Asian, "-ville" French or American)

  python probe.py --tag pythia-410m@8000
  python probe.py --spell
  python probe.py --replicate pythia-1b@143000.title    # lat/lon R^2, Godey et al.'s split

Writes results/preds/<tag>.parquet (pid, fold, layer, pred_lat, pred_lon, err_km,
null_km) and results/preds/<tag>.layers.csv (out-of-fold error at every layer).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

import config as C
import geo

ALPHAS = np.logspace(-1, 5, 13)


def places():
    return pd.read_parquet(os.path.join(C.DATA, "places.parquet"))


def load_states(tag):
    z = np.load(os.path.join(C.STATES, tag + ".npz"))
    return z["last"], z["layers"], z["pid"]


def standardise(tr, te):
    mu = tr.mean(0)
    sd = tr.std(0)
    sd[sd < 1e-6] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


class FastRidgeCV:
    """Ridge with an intercept and the penalty chosen by exact leave-one-out error, the
    criterion of sklearn's RidgeCV, computed from one eigendecomposition of X'X.
    With X centred, the LOO residual of place i is (y_i - yhat_i) / (1 - h_i), where
    h_i = 1/n + sum_k z_ik^2 / (lam_k + alpha) and z = X V. Everything is matrix
    products, several times faster than RidgeCV's SVD for wide hidden states."""

    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    def fit(self, X, Y):
        X = np.asarray(X, np.float64)
        self.x_mean = X.mean(0)
        self.y_mean = Y.mean(0)
        Xc, Yc = X - self.x_mean, Y - self.y_mean
        lam, V = np.linalg.eigh(Xc.T @ Xc)
        lam = np.clip(lam, 0, None)
        Z = Xc @ V
        Z2 = Z * Z
        ZtY = Z.T @ Yc
        n = len(X)
        best = None
        for a in self.alphas:
            d = 1.0 / (lam + a)
            fit = Z @ (d[:, None] * ZtY)
            h = 1.0 / n + Z2 @ d
            mse = np.mean(((Yc - fit) / (1 - h)[:, None]) ** 2)
            if best is None or mse < best[0]:
                best = (mse, a, d)
        self.best_score_, self.alpha_ = -best[0], best[1]
        self.coef_ = V @ (best[2][:, None] * ZtY)          # (p, targets)
        return self

    def predict(self, X):
        return (np.asarray(X, np.float64) - self.x_mean) @ self.coef_ + self.y_mean


def fit_layer(X, Y, folds):
    """Out-of-fold predictions for one layer, and each training fold's LOO error."""
    X = X.astype(np.float32)
    pred = np.zeros_like(Y)
    loo = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, Xte = standardise(X[tr], X[te])
        m = FastRidgeCV().fit(Xtr, Y[tr])
        pred[te] = m.predict(Xte)
        loo[f] = -m.best_score_
    return pred, loo


def probe(X3, Y, folds, jobs=3):
    """X3: (n, layers, d). Returns per-layer predictions, chosen layer per fold, final."""
    res = Parallel(n_jobs=jobs)(delayed(fit_layer)(X3[:, j], Y, folds) for j in range(X3.shape[1]))
    preds = np.stack([p for p, _ in res], 1)                      # (n, layers, 3)
    final = np.zeros_like(Y)
    chosen = np.zeros(len(Y), int)
    for f in np.unique(folds):
        j = int(np.argmin([loo[f] for _, loo in res]))
        te = folds == f
        final[te] = preds[te, j]
        chosen[te] = j
    return preds, chosen, final


def null_pred(Y, folds):
    out = np.zeros_like(Y)
    for f in np.unique(folds):
        out[folds == f] = Y[folds != f].mean(0)
    return out


def spell_pred(names, Y, folds):
    """Character n-gram ridge; penalty chosen on a held-out fifth of each training fold."""
    rng = np.random.default_rng(C.SEED)
    out = np.zeros_like(Y)
    for f in np.unique(folds):
        tr = np.flatnonzero(folds != f)
        te = np.flatnonzero(folds == f)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2,
                              sublinear_tf=True, lowercase=True)
        Xtr = vec.fit_transform([names[i] for i in tr])
        Xte = vec.transform([names[i] for i in te])
        inner = rng.random(len(tr)) < 0.2
        best = None
        for a in [0.3, 1, 3, 10, 30]:
            m = Ridge(alpha=a, solver="sparse_cg").fit(Xtr[~inner], Y[tr][~inner])
            e = np.median(geo.gc_km(m.predict(Xtr[inner]), Y[tr][inner]))
            if best is None or e < best[0]:
                best = (e, a)
        out[te] = Ridge(alpha=best[1], solver="sparse_cg").fit(Xtr, Y[tr]).predict(Xte)
    return out


def save(tag, df, pred, null, layer, layers_csv=None):
    os.makedirs(C.PREDS, exist_ok=True)
    lat, lon = geo.to_latlon(pred)
    Y = geo.to_xyz(df.lat.values, df.lon.values)
    out = pd.DataFrame(dict(pid=df.pid.values, fold=df.fold.values, layer=layer,
                            pred_lat=lat, pred_lon=lon, err_km=geo.gc_km(pred, Y),
                            null_km=geo.gc_km(null, Y)))
    out.to_parquet(os.path.join(C.PREDS, tag + ".parquet"), index=False)
    if layers_csv is not None:
        layers_csv.to_csv(os.path.join(C.PREDS, tag + ".layers.csv"), index=False)
    return out


def run_tag(tag, jobs):
    df = places()
    X3, layers, pid = load_states(tag)
    assert (pid == df.pid.values).all(), "states and places out of step"
    Y = geo.to_xyz(df.lat.values, df.lon.values)
    folds = df.fold.values
    t0 = time.time()
    preds, chosen, final = probe(X3, Y, folds, jobs)
    null = null_pred(Y, folds)
    rows = []
    for j, l in enumerate(layers):
        e = geo.gc_km(preds[:, j], Y)
        rows.append(dict(layer=int(l), median_km=np.median(e), mean_km=e.mean(),
                         r2=r2_score(Y, preds[:, j])))
    lay = pd.DataFrame(rows)
    out = save(tag, df, final, null, layers[chosen], lay)
    print("{}: median error {:,.0f} km (null {:,.0f}), layers chosen {}, {:.1f} min".format(
        tag, out.err_km.median(), out.null_km.median(), sorted(set(layers[chosen].tolist())),
        (time.time() - t0) / 60), flush=True)


def run_spell():
    df = places()
    Y = geo.to_xyz(df.lat.values, df.lon.values)
    pred = spell_pred(df.name.tolist(), Y, df.fold.values)
    out = save("spell", df, pred, null_pred(Y, df.fold.values), -1)
    print("spelling: median error {:,.0f} km".format(out.err_km.median()))


def replicate(tag, jobs):
    """Godey et al. (2024): full titles, all places, their 80/20 split, ridge on lat and
    lon, R^2 averaged over the two. Best layer on the test set, as in their tables."""
    allp = pd.read_parquet(os.path.join(C.DATA, "places_all.parquet"))
    X3, layers, pid = load_states(tag)
    assert (pid == allp.pid.values).all()
    test = allp.is_test.values.astype(bool)
    T = allp[["lat", "lon"]].values

    def one(j):
        Xtr, Xte = standardise(X3[~test, j].astype(np.float32), X3[test, j].astype(np.float32))
        m = FastRidgeCV().fit(Xtr, T[~test])
        p = m.predict(Xte)
        return (r2_score(T[test, 0], p[:, 0]) + r2_score(T[test, 1], p[:, 1])) / 2

    r2 = Parallel(n_jobs=jobs)(delayed(one)(j) for j in range(X3.shape[1]))
    out = pd.DataFrame(dict(layer=layers, r2=r2))
    os.makedirs(C.PREDS, exist_ok=True)
    out.to_csv(os.path.join(C.PREDS, tag + ".replicate.csv"), index=False)
    print("{}: best R^2 {:.1f} at layer {} of {}".format(
        tag, 100 * out.r2.max(), int(out.layer[out.r2.idxmax()]), int(layers.max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--spell", action="store_true")
    ap.add_argument("--replicate")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    if args.spell:
        run_spell()
    if args.tag:
        run_tag(args.tag, args.jobs)
    if args.replicate:
        replicate(args.replicate, args.jobs)


if __name__ == "__main__":
    main()
