"""The places a model is asked about, with true coordinates, country, region and fold.

Source: the world places table of Gurnee and Tegmark (2023), 39,585 Wikipedia places
(cities, regions, buildings, lakes, mountains) with coordinates and page views.

Cleaning:
  1. The Wikipedia title often names the location ("Burgdorf, Switzerland", "Leyburn,
     Queensland", "San José (canton)"). The probe would read the country off the text,
     so the prompt name is the title cut at the first ", " or " (".
  2. Cut names that now collide with another place ("Delaware County" for Ohio and
     Pennsylvania) are dropped altogether: there is no way to know which one a model
     would think of.
  3. Country and region come from the coordinates (Natural Earth 1:10m countries), not
     the table's own country field, which mixes "England", "United_Kingdom",
     "Hokkaido" and "List_of_World_Heritage_Sites_in_the_Philippines". Points just off
     a coastline take the nearest country within 30 km.
  4. Each country keeps at most CAP places, chosen at random, so the probe is fit on a
     world rather than on the United States, Britain, India and Australia.
  5. Folds are assigned within each country, so every country is spread over all folds.

Writes data/places.parquet (the sample) and data/places_all.parquet (every place with
its full title and the original train/test split, for the replication of prior work).
"""
import os
import re

import cartopy.io.shapereader as shp
import geopandas as gpd
import numpy as np
import pandas as pd
import requests

import config as C

SUFFIX = re.compile(r", | \(")


def base_name(title):
    return SUFFIX.split(title, maxsplit=1)[0].strip()


def load_raw():
    path = os.path.join(C.RAW, "world_place.csv")
    if not os.path.exists(path):
        os.makedirs(C.RAW, exist_ok=True)
        r = requests.get(C.PLACES_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return pd.read_csv(path)


def countries():
    path = shp.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
    g = gpd.read_file(path)[["ADMIN", "ADM0_A3", "NAME", "CONTINENT", "SUBREGION",
                             "POP_EST", "geometry"]]
    return g.rename(columns={"ADMIN": "country", "ADM0_A3": "iso", "NAME": "short",
                             "CONTINENT": "continent", "SUBREGION": "subregion",
                             "POP_EST": "country_pop"})


def locate(df, polys):
    """Country of each point: containing polygon, else nearest within 30 km."""
    pts = gpd.GeoDataFrame(df[["pid"]], geometry=gpd.points_from_xy(df.longitude, df.latitude),
                           crs="EPSG:4326")
    polys = polys.to_crs("EPSG:4326")
    hit = gpd.sjoin(pts, polys, how="left", predicate="within").drop(columns="index_right")
    hit = hit[~hit.pid.duplicated()]
    miss = hit[hit.iso.isna()][["pid", "geometry"]]
    if len(miss):
        eq = "EPSG:8857"    # Equal Earth, metres
        near = gpd.sjoin_nearest(miss.to_crs(eq), polys.to_crs(eq), how="left",
                                 max_distance=30000).drop(columns="index_right")
        near = near[~near.pid.duplicated()].set_index("pid")
        cols = [c for c in polys.columns if c != "geometry"]
        hit = hit.set_index("pid")
        hit.loc[near.index, cols] = near[cols]
        hit = hit.reset_index()
    return pd.DataFrame(hit.drop(columns="geometry"))


def assign_folds(df, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), int)
    for _, idx in df.groupby("iso").indices.items():
        idx = rng.permutation(idx)
        fold[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return fold


def main():
    raw = load_raw()
    raw["pid"] = np.arange(len(raw))
    raw["title"] = raw.name.astype(str).str.strip()
    raw["name"] = raw.title.map(base_name)
    raw["had_suffix"] = raw.name != raw.title
    raw = raw.rename(columns={"latitude": "lat", "longitude": "lon", "country": "country_raw"})
    raw = raw[raw.lat.between(-90, 90) & raw.lon.between(-180, 180)]

    loc = locate(raw.rename(columns={"lat": "latitude", "lon": "longitude"}), countries())
    df = raw.merge(loc, on="pid", how="left")
    df["region"] = df.subregion.map(C.REGION_OF)

    # every place, full titles, original split: for the replication of Godey et al.
    df.to_parquet(os.path.join(C.DATA, "places_all.parquet"), index=False)

    n0 = len(df)
    log = {"raw": n0}
    df = df[df.region.notna()]
    log["in a region"] = len(df)
    df = df[(df.name.str.len() >= 3) & ~df.name.str.fullmatch(r"[\d\W]+")]
    log["usable name"] = len(df)
    key = df.name.str.casefold()
    df = df[~key.duplicated(keep=False)]
    log["unique cut name"] = len(df)

    rng = np.random.default_rng(C.SEED)
    keep = []
    for _, g in df.groupby("iso"):
        keep.append(g if len(g) <= C.CAP else g.iloc[rng.choice(len(g), C.CAP, replace=False)])
    df = pd.concat(keep).sort_values("pid").reset_index(drop=True)
    log["after cap"] = len(df)
    df["fold"] = assign_folds(df, C.FOLDS, C.SEED)

    cols = ["pid", "name", "title", "had_suffix", "lat", "lon", "iso", "country", "short",
            "continent", "subregion", "region", "country_pop", "entity_type",
            "entity_subtype", "page_views", "population", "country_raw", "fold"]
    df[cols].to_parquet(os.path.join(C.DATA, "places.parquet"), index=False)
    for k, v in log.items():
        print("{:>18}: {:,}".format(k, v))
    print(df.region.value_counts().to_string())
    print("countries:", df.iso.nunique())
    print(df.groupby("region").iso.nunique().to_string())


if __name__ == "__main__":
    main()
