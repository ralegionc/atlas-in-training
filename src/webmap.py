"""The interactive map: docs/data.js for docs/index.html.

A slider runs through the tracked checkpoints of either model. Two views: every place at
its true position lit by how close the probe puts it, or every place at the probe's
predicted position. Searching a name draws that place's path through training, from the
middle of the cloud to wherever the model ends up putting it.

Data are packed as base64 typed arrays per model and checkpoint: predicted latitude and
longitude in hundredths of a degree (int16), brightness in 0..255 (uint8,
config.brightness of the error) and the error in tens of km (uint16).
"""
import base64
import json
import os

import numpy as np

import analysis as A
import config as C


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(size, steps, pl):
    long = A.tracked(size, steps)
    long = long[long.pid.isin(pl.pid)]
    steps = sorted(long.step.unique())
    pos = {p: i for i, p in enumerate(pl.pid)}
    n, T = len(pl), len(steps)
    plat = np.zeros((T, n), np.int16)
    plon = np.zeros((T, n), np.int16)
    know = np.zeros((T, n), np.uint8)
    err = np.zeros((T, n), np.uint16)
    med = []
    for t, s in enumerate(steps):
        d = long[long.step == s]
        idx = d.pid.map(pos).values
        plat[t, idx] = np.round(d.pred_lat.values * 100)
        plon[t, idx] = np.round(d.pred_lon.values * 100)
        know[t, idx] = np.round(C.brightness(d.err_km.values) * 255)
        err[t, idx] = np.round(np.minimum(d.err_km.values, 20000) / 10)
        med.append(float(d.err_km.median()))
    return dict(steps=[int(s) for s in steps], median_km=med, plat=b64(plat), plon=b64(plon),
                know=b64(know), err=b64(err))


def build():
    pl = A.places().sort_values("pid").reset_index(drop=True)
    ct = A.counts()
    models = {}
    for size, steps in [(C.BIG_MODEL, C.BIG_STEPS), (C.TRACK_MODEL, C.TRACK_STEPS)]:
        models[size] = pack(size, steps, pl)
    cnt = None
    if ct is not None:
        cnt = pl[["pid"]].merge(ct[["pid", "count"]], on="pid", how="left")["count"].fillna(-1).astype(int).tolist()
    payload = dict(models=models, order=list(models), tokens_per_step=C.TOKENS_PER_STEP,
                   bright_km=list(C.BRIGHT_KM), names=pl.name.tolist(), country=pl.country.tolist(),
                   region=pl.region.tolist(), lat=np.round(pl.lat.values, 2).tolist(),
                   lon=np.round(pl.lon.values, 2).tolist(), count=cnt)
    os.makedirs(C.DOCS, exist_ok=True)
    with open(os.path.join(C.DOCS, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.ATLAS = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print("docs/data.js: {:,} places; {}".format(
        len(pl), ", ".join("{} x {}".format(k, len(v["steps"])) for k, v in models.items())))


if __name__ == "__main__":
    build()
