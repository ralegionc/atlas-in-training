"""Names that belong to more than one real place.

A model asked about "Victoria" or "Cambridge" may think of a different, more famous
place than the one in the table, and the Pile count of the name counts them all. Each
sample place is matched by name (case-folded, against GeoNames' name and ASCII name)
to every GeoNames town of 1,000 people or more; a homonym is one more than 300 km away.

The Pile count of a name that is also an ordinary word ("Tyre", "Bath", "Reading",
"Mobile") mostly counts the word. A name is word-like if WordNet has a sense of it that
is not a named instance (a city, a person): "tyre" has the car part, "Burgdorf" nothing.

Source: GeoNames cities1000 (download.geonames.org/export/dump/cities1000.zip), WordNet.
Writes data/homonyms.parquet: pid, n_far (homonyms > 300 km away), far_pop (the
largest one's population), wordlike.
"""
import os
import zipfile

import numpy as np
import pandas as pd
import requests

import config as C
import geo

URL = "https://download.geonames.org/export/dump/cities1000.zip"
COLS = ["gid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2",
        "a1", "a2", "a3", "a4", "population", "elev", "dem", "tz", "mod"]


def load():
    d = os.path.join(C.RAW, "geonames")
    txt = os.path.join(d, "cities1000.txt")
    if not os.path.exists(txt):
        os.makedirs(d, exist_ok=True)
        z = os.path.join(d, "cities1000.zip")
        with open(z, "wb") as f:
            f.write(requests.get(URL, timeout=120).content)
        zipfile.ZipFile(z).extractall(d)
    return pd.read_csv(txt, sep="\t", names=COLS, usecols=["name", "asciiname", "lat", "lon",
                                                             "population"],
                       quoting=3, keep_default_na=False, dtype={"name": str, "asciiname": str})


def wordlike(name):
    from nltk.corpus import wordnet as wn
    return any(not s.instance_hypernyms() for s in wn.synsets(name.lower().replace(" ", "_")))


def main():
    gn = load()
    keys = pd.concat([gn[["name", "lat", "lon", "population"]],
                      gn[["asciiname", "lat", "lon", "population"]].rename(columns={"asciiname": "name"})])
    keys["key"] = keys.name.str.casefold()
    keys = keys.drop_duplicates(["key", "lat", "lon"])
    pl = pd.read_parquet(os.path.join(C.DATA, "places.parquet"))
    pl["key"] = pl.name.str.casefold()
    m = pl[["pid", "key", "lat", "lon"]].merge(keys, on="key", suffixes=("", "_g"))
    m["km"] = geo.gc_km(geo.to_xyz(m.lat.values, m.lon.values), geo.to_xyz(m.lat_g.values, m.lon_g.values))
    far = m[m.km > 300]
    out = pl[["pid"]].merge(far.groupby("pid").agg(n_far=("km", "size"), far_pop=("population", "max")),
                            on="pid", how="left").fillna({"n_far": 0, "far_pop": 0})
    out = out.merge(pl[["pid", "name"]], on="pid")
    out["wordlike"] = out.name.map(wordlike)
    out = out.drop(columns="name")
    out.to_parquet(os.path.join(C.DATA, "homonyms.parquet"), index=False)
    print("word-like names: {:.1%}".format(out.wordlike.mean()))
    print("places with a far homonym: {:.1%}".format((out.n_far > 0).mean()))
    print(pl.merge(out, on="pid").query("n_far > 0").sort_values("far_pop", ascending=False)
          [["name", "country", "n_far", "far_pop"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
