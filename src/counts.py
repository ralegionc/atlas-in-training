"""How often the Pile mentions each place, and how often it mentions it near its country.

Uses the infini-gram API (api.infini-gram.io), an exact index of the Pile's training
split, the data Pythia was trained on. For each place:

  count     occurrences of the name
  own       the name within WINDOW tokens of its own country ("Burgdorf" ... "Switzerland")
  placebo   the same for another place's country: countries are shuffled among places,
            so placebo countries are as big and as often named as the real ones

and once per country, how often the country is named at all (the sum over its names).

A clause with more than MAX_CLAUSE matches is subsampled by the engine (approx=True),
so co-mentions of very common names are estimates. Results are cached line by line in
data/ig_cache.jsonl, so the script resumes where it stopped.

  python counts.py [--only count] [--interval 1.1]

Writes data/counts.parquet and data/country_counts.parquet.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cartopy.io.shapereader as shp
import geopandas as gpd
import numpy as np
import pandas as pd
import requests

import config as C

URL = "https://api.infini-gram.io/"
MAX_CLAUSE = 500000

# How each country is named. Natural Earth's short, long and alternative names, minus
# the abbreviated map labels ("Dem. Rep. Korea"), unless set here. The API takes at
# most four alternatives per clause.
NAMES = {
    "USA": ["United States", "USA", "U.S.", "America"],
    "GBR": ["United Kingdom", "England", "Scotland", "Wales"],
    "KOR": ["South Korea", "Korea"],
    "PRK": ["North Korea"],
    "CZE": ["Czech Republic", "Czechia"],
    "SWZ": ["Eswatini", "Swaziland"],
    "MKD": ["North Macedonia", "Macedonia"],
    "MMR": ["Myanmar", "Burma"],
    "NLD": ["Netherlands", "Holland"],
    "COD": ["Democratic Republic of the Congo", "DR Congo", "Congo"],
    "COG": ["Republic of the Congo", "Congo"],
    "CIV": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"],
    "TLS": ["East Timor", "Timor-Leste"],
    "CPV": ["Cape Verde", "Cabo Verde"],
    "PSX": ["Palestine", "West Bank", "Gaza"],
    "VNM": ["Vietnam", "Viet Nam"],
    "FRO": ["Faroe Islands", "Faroes"],
    "MAC": ["Macau", "Macao"],
    "BIH": ["Bosnia and Herzegovina", "Bosnia"],
    "TUR": ["Turkey"],
    "LAO": ["Laos"],
    "BRN": ["Brunei"],
    "FSM": ["Micronesia"],
    "SDS": ["South Sudan"],
    "CAF": ["Central African Republic"],
    "DOM": ["Dominican Republic"],
    "GNQ": ["Equatorial Guinea"],
    "SAH": ["Western Sahara"],
    "CYN": ["Northern Cyprus"],
    "CNM": ["Cyprus"],
}


def country_names():
    g = gpd.read_file(shp.natural_earth(resolution="10m", category="cultural",
                                        name="admin_0_countries"))
    out = {}
    for _, r in g.iterrows():
        if r.ADM0_A3 in NAMES:
            out[r.ADM0_A3] = NAMES[r.ADM0_A3]
            continue
        s = []
        for k in ["NAME_LONG", "NAME", "NAME_ALT"]:
            v = r[k]
            if isinstance(v, str) and v and "." not in v and v not in s:
                s.append(v)
        out[r.ADM0_A3] = s[:4]
    return out


def placebo_iso(iso, seed=C.SEED):
    """Each place gets another place's country: a shuffle with no fixed points left."""
    rng = np.random.default_rng(seed)
    iso = np.asarray(iso)
    out = iso[rng.permutation(len(iso))]
    for i in np.flatnonzero(out == iso):
        # swap with a random place whose country differs from both: one swap at a time,
        # so the multiset of countries is untouched
        while out[i] == iso[i]:
            j = rng.integers(len(iso))
            if out[j] != iso[i] and iso[j] != out[i]:
                out[i], out[j] = out[j], out[i]
    assert (out != iso).all()
    return out


def clause(names):
    return " OR ".join(names)


class Engine:
    def __init__(self, index, path):
        self.index = index
        self.path = path
        self.cache = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:    # a line cut short by a kill
                        continue
                    self.cache[r["key"]] = r
        self.lock = threading.Lock()
        self.local = threading.local()     # one keep-alive session per worker thread
        self.interval = 1.1
        self.next_at = 0.0

    def throttle(self):
        with self.lock:
            now = time.time()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))

    def count(self, query, cnf=False):
        key = query + ("|cnf" if cnf else "")
        if key in self.cache:
            return self.cache[key]
        payload = dict(index=self.index, query_type="count", query=query)
        if cnf:
            payload.update(max_clause_freq=MAX_CLAUSE, max_diff_tokens=C.WINDOW)
        r = None
        for attempt in range(12):
            self.throttle()
            try:
                if not hasattr(self.local, "s"):
                    self.local.s = requests.Session()
                resp = self.local.s.post(URL, json=payload, timeout=120)
                if resp.status_code == 403:
                    raise PermissionError("rate limited")
                r = resp.json()
                if "error" in r:
                    raise RuntimeError(r["error"])
                if "count" not in r:              # a malformed reply: try again
                    raise ConnectionError(str(r)[:200])
                break
            except Exception as e:
                if attempt == 11 or (isinstance(e, RuntimeError) and attempt >= 2):
                    print("giving up on", repr(query), e, flush=True)
                    return None
                wait = 300 if isinstance(e, PermissionError) else 2 ** attempt
                print("  {} ({}), waiting {} s".format(type(e).__name__, e, wait), flush=True)
                time.sleep(wait)
        rec = dict(key=key, count=r["count"], approx=bool(r.get("approx", False)),
                   ntok=len(r.get("tokens") or []))
        with self.lock:
            self.cache[key] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="count,own,placebo")
    ap.add_argument("--interval", type=float, default=1.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--placebo-n", type=int, default=6000,
                    help="places that get a placebo query (the API allows about 1 query/s)")
    ap.add_argument("--offline", action="store_true", help="build tables from the cache only")
    args = ap.parse_args()
    eng = Engine(C.INDEX, os.path.join(C.DATA, "ig_cache.jsonl"))
    eng.interval = args.interval

    df = pd.read_parquet(os.path.join(C.DATA, "places.parquet"))
    names = country_names()
    df["placebo_iso"] = placebo_iso(df.iso.values)
    isos = sorted(set(df.iso) | set(df.placebo_iso))

    # countries first (a few hundred queries), then places in random order, pass by pass
    order = df.sample(frac=1, random_state=C.SEED)
    query = {"count": lambda r: (r.name, False),
             "own": lambda r: ("{} AND {}".format(r.name, clause(names[r.iso])), True),
             "placebo": lambda r: ("{} AND {}".format(r.name, clause(names[r.placebo_iso])), True)}
    kinds = args.only.split(",")
    jobs = [(None, "country", n, False) for i in isos for n in names[i]]
    for k in kinds:
        rows = order.head(args.placebo_n) if k == "placebo" else order
        jobs += [(r.pid, k) + query[k](r) for r in rows.itertuples()]
    key = lambda q, cnf: q + ("|cnf" if cnf else "")
    todo = [] if args.offline else [j for j in jobs if key(j[2], j[3]) not in eng.cache]
    print("{:,} queries, {:,} not cached".format(len(jobs), len(todo)), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(eng.count, q, cnf) for _, _, q, cnf in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 500 == 0:
                rate = i / (time.time() - t0)
                print("{:,}/{:,}  {:.2f}/s  eta {:.0f} min".format(
                    i, len(todo), rate, (len(todo) - i) / rate / 60), flush=True)

    get = lambda q, cnf: eng.cache.get(key(q, cnf))
    cc = []
    for i in isos:
        recs = [get(n, False) for n in names[i]]
        cc.append(dict(iso=i, names=" | ".join(names[i]),
                       country_count=sum(r["count"] for r in recs) if all(recs) else None))
    cc = pd.DataFrame(cc)
    cc.to_parquet(os.path.join(C.DATA, "country_counts.parquet"), index=False)

    out = df[["pid", "name", "iso", "placebo_iso"]].copy()
    for k in kinds:
        recs = [get(*query[k](r)) for r in df.itertuples()]
        out[k] = [r["count"] if r else None for r in recs]
        out[k + "_approx"] = [r["approx"] if r else None for r in recs]
        if k == "count":
            out["ntok_llama"] = [r["ntok"] if r else None for r in recs]
    out = out.merge(cc[["iso", "country_count"]], on="iso", how="left")
    out = out.merge(cc[["iso", "country_count"]].rename(
        columns={"iso": "placebo_iso", "country_count": "placebo_country_count"}),
        on="placebo_iso", how="left")
    out.to_parquet(os.path.join(C.DATA, "counts.parquet"), index=False)
    print(out.describe().to_string())


if __name__ == "__main__":
    main()
