"""Known-answer tests: geometry, name cleaning, the placebo shuffle, and probes that must
recover a planted map and must not find one in noise."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import counts  # noqa: E402
import geo  # noqa: E402
import places  # noqa: E402
import probe  # noqa: E402


def test_roundtrip():
    lat = np.array([0, 45, -33.9, 89.9, -89.9, 10])
    lon = np.array([0, 179.9, 151.2, -120, 60, -179.9])
    la, lo = geo.to_latlon(geo.to_xyz(lat, lon))
    assert np.allclose(la, lat, atol=1e-9)
    assert np.allclose(lo, lon, atol=1e-9)


def test_distances():
    london, paris = geo.to_xyz(51.5074, -0.1278), geo.to_xyz(48.8566, 2.3522)
    assert geo.gc_km(london, paris) == pytest.approx(343.5, abs=2)
    # across the date line is short, not half the world
    a, b = geo.to_xyz(0, 179.5), geo.to_xyz(0, -179.5)
    assert geo.gc_km(a, b) == pytest.approx(111.2, abs=0.5)
    # antipodes, and a shrunk prediction scores by direction only
    assert geo.gc_km(geo.to_xyz(0, 0), geo.to_xyz(0, 180)) == pytest.approx(np.pi * 6371, rel=1e-6)
    assert geo.gc_km(0.2 * london, paris) == pytest.approx(343.5, abs=2)


def test_base_name():
    assert places.base_name("Burgdorf, Switzerland") == "Burgdorf"
    assert places.base_name("San José (canton)") == "San José"
    assert places.base_name("Gainesville, Georgia") == "Gainesville"
    assert places.base_name("Burton-in-Kendal") == "Burton-in-Kendal"
    assert places.base_name("Dongcheng District, Beijing") == "Dongcheng District"


def test_placebo_shuffle():
    rng = np.random.default_rng(1)
    iso = rng.choice(["USA", "GBR", "IND", "FJI", "BOL"], 5000, p=[.4, .3, .2, .05, .05])
    out = counts.placebo_iso(iso)
    assert (out != iso).all()
    assert sorted(out) == sorted(iso)       # same countries, as often


def world(n, seed):
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-0.9, 0.95, n)))
    lon = rng.uniform(-180, 180, n)
    return lat, lon, geo.to_xyz(lat, lon), rng


def test_probe_recovers_planted_map():
    """States = random rotation of the location (in 3 of 64 dims) plus noise: the probe
    must put places within a few hundred km, and must pick the informative layer."""
    lat, lon, Y, rng = world(3000, 0)
    d = 64
    W = rng.normal(size=(3, d))
    noise = rng.normal(size=(len(Y), 2, d))
    X = np.stack([noise[:, 0], Y @ W + 0.05 * noise[:, 1]], 1)   # layer 0 noise, layer 1 signal
    folds = np.arange(len(Y)) % 5
    _, chosen, final = probe.probe(X, Y, folds, jobs=1)
    err = geo.gc_km(final, Y)
    assert (chosen == 1).all()
    assert np.median(err) < 300


def test_probe_finds_nothing_in_noise():
    lat, lon, Y, rng = world(3000, 1)
    X = rng.normal(size=(len(Y), 1, 64))
    folds = np.arange(len(Y)) % 5
    _, _, final = probe.probe(X, Y, folds, jobs=1)
    err = np.median(geo.gc_km(final, Y))
    null = np.median(geo.gc_km(probe.null_pred(Y, folds), Y))
    assert err > 0.97 * null


def test_spelling_baseline_learns_suffixes():
    """Names ending in -pur sit in India, -ville in France: the n-gram model must learn it,
    and must not do better than chance on random strings."""
    rng = np.random.default_rng(2)
    letters = list("abcdefghijklmnoprstuvz")
    stems = ["".join(rng.choice(letters, 6)) for _ in range(1200)]
    names = [s + ("pur" if i % 2 else "ville") for i, s in enumerate(stems)]
    lat = np.where(np.arange(1200) % 2, 22.0, 47.0) + rng.normal(0, 2, 1200)
    lon = np.where(np.arange(1200) % 2, 78.0, 2.0) + rng.normal(0, 2, 1200)
    Y = geo.to_xyz(lat, lon)
    folds = np.arange(1200) % 5
    err = np.median(geo.gc_km(probe.spell_pred(names, Y, folds), Y))
    assert err < 800
    shuffled = probe.spell_pred(stems, Y, folds)    # stems carry no information
    assert np.median(geo.gc_km(shuffled, Y)) > 2500


def test_fast_ridge_matches_sklearn():
    """Same penalty, same coefficients, same LOO score as sklearn's RidgeCV."""
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 80))
    Y = X[:, :3] @ rng.normal(size=(3, 3)) + rng.normal(size=(600, 3))
    for scale in [1.0, 0.05]:   # strong and weak signal pick different penalties
        Ys = Y * scale + rng.normal(size=Y.shape)
        a = probe.FastRidgeCV().fit(X, Ys)
        b = RidgeCV(alphas=probe.ALPHAS).fit(X, Ys)
        assert a.alpha_ == b.alpha_
        assert np.allclose(a.predict(X), b.predict(X), atol=1e-6)
        assert a.best_score_ == pytest.approx(b.best_score_, rel=1e-6)


def test_exposure_alpha_recovers_planted():
    """Error that depends on 0.5 log count + log step must give alpha near 0.5, and error
    that ignores the count must give alpha near 0."""
    import pandas as pd
    import analysis
    rng = np.random.default_rng(4)
    n = 3000
    lc = rng.uniform(0, 4, n)
    steps = np.array([1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000])
    rows = []
    for a_true in (0.5, 0.0):
        for s in steps:
            z = a_true * lc + np.log10(s)
            y = 3.6 - 0.25 * z + rng.normal(0, 0.05, n)
            rows.append(pd.DataFrame(dict(pid=np.arange(n), y=y, lc=lc, ls=np.log10(s), a=a_true)))
    d = pd.concat(rows)
    for a_true in (0.5, 0.0):
        a, _, _ = analysis.exposure_alpha(d[d.a == a_true])
        assert abs(a - a_true) < 0.08


def test_log_interp_step():
    import analysis
    steps = np.array([1.0, 10.0, 100.0, 1000.0])
    vals = np.array([0.0, 0.2, 0.6, 1.0])
    # half way between 0.2 at 10 and 0.6 at 100, in log step: sqrt(10 * 100)
    assert analysis.log_interp_step(steps, vals, 0.4) == pytest.approx(10 ** 1.5)
    assert np.isnan(analysis.log_interp_step(steps, vals, 2.0))
