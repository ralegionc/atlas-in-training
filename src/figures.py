"""Figures. Dark posters in the style of the sibling projects: one warm hue glowing on a
night background, real coastlines as a faint lace underneath.

  fig1_hero.png      the map forming: each place at the probe's predicted position, and
                     for each region an arrow from where it is to where the model puts it
  fig2_known.png     each place at its true position, lit by how close the probe puts it
  fig3_regions.png   error over training in each region, against the spelling floor
  fig4_beyond.png    what the model adds beyond spelling, through training, by region, for
                     both tracked models
  fig5_counts.png    final error against how often the Pile names a place, within country
  fig6_exposure.png  error over training by how often a place is named: against steps,
                     then against mentions seen so far
  fig7_scaling.png   final error by model size, with the spelling floor
  learning.gif       the hero map through every checkpoint

  python figures.py [--only hero,known,regions,beyond,counts,exposure,scaling,gif] [--size 1.4b]
"""
import argparse
import os

import cartopy.crs as ccrs
import cartopy.io.shapereader as shp
import geopandas as gpd
import matplotlib
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import analysis as A  # noqa: E402
import config as C  # noqa: E402
import geo  # noqa: E402

BG = "#0a0d12"
TEXT, SUBTLE, FAINT = "#e8e4da", "#8a8f98", "#5a616c"
LACE = "#252c37"
AMBER = "#f6b73c"
ICE = "#a9cdee"
GLOW = LinearSegmentedColormap.from_list("glow", ["#15191f", "#3a2f1c", "#8a5f1d", "#d9952c", AMBER, "#fff2cf"])
# ordinal ramp for count bins: dim to bright, one hue
BINS = ["#6b4a1a", "#9a6a22", "#c98a2a", "#f0ad3a", "#ffe0a0"]
FONT = "DejaVu Sans"
PROJ = ccrs.EqualEarth()
PC = ccrs.PlateCarree()
HERO_STEPS = [0, 1000, 4000, 16000, 45000, 143000]
# regions that get an arrow, and their label
ARROWS = {"North America": "North America", "Central America & Caribbean": "Central America",
          "South America": "South America", "Northern & Western Europe": "NW Europe",
          "Southern & Eastern Europe": "SE Europe", "Middle East & North Africa": "Middle East",
          "Sub-Saharan Africa": "Africa", "South Asia": "South Asia", "East Asia": "East Asia",
          "Southeast Asia": "SE Asia", "Australia & New Zealand": "Australia & NZ",
          "Pacific Islands": "Pacific", "Central Asia": "Central Asia"}
HALO = [pe.withStroke(linewidth=2.5, foreground=BG)]

plt.rcParams.update({"font.family": FONT, "text.color": TEXT, "axes.labelcolor": SUBTLE,
                     "xtick.color": SUBTLE, "ytick.color": SUBTLE, "axes.edgecolor": FAINT,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False})


def land():
    return gpd.read_file(shp.natural_earth(resolution="50m", category="physical", name="land"))


def world_ax(fig, rect, lnd):
    ax = fig.add_axes(rect, projection=PROJ)
    ax.set_global()
    ax.add_geometries(lnd.geometry, crs=PC, facecolor="#10141a", edgecolor=LACE,
                      linewidth=0.35, zorder=0)
    ax.spines["geo"].set_edgecolor(LACE)
    ax.spines["geo"].set_linewidth(0.6)
    ax.set_facecolor(BG)
    return ax


def step_label(s):
    return "initialisation" if s == 0 else "step {:,}".format(int(s))


def tokens_label(s):
    t = s * C.TOKENS_PER_STEP
    return "0 tokens" if s == 0 else ("{:.1f}B tokens".format(t / 1e9) if t >= 1e9
                                      else "{:.0f}M tokens".format(t / 1e6))


def log_step_axis(ax, steps):
    """x axis in log step with initialisation drawn at a gap to the left."""
    ax.set_xscale("log")
    ax.set_xlim(40, 2.2e5)
    ticks = [1000, 10000, 100000]
    ax.set_xticks([60] + ticks)
    ax.set_xticklabels(["init", "1k", "10k", "100k"])
    ax.tick_params(axis="x", which="minor", length=0)


def xstep(s):
    return np.where(np.asarray(s) == 0, 60, s)


def count_brightness(ct, pids):
    if ct is None:
        return np.full(len(pids), 0.6)
    c = ct.set_index("pid")["count"].reindex(pids).fillna(0).values
    lc = np.log10(c + 1)
    return np.clip(0.15 + 0.85 * lc / np.percentile(lc, 99), 0, 1)


def glow_scatter(ax, lon, lat, val, size=2.2, alpha=0.9):
    o = np.argsort(val)
    lon, lat, val = np.asarray(lon)[o], np.asarray(lat)[o], np.asarray(val)[o]
    ax.scatter(lon, lat, s=size * 5 * (0.2 + val), c=val, cmap=GLOW, vmin=0, vmax=1,
               alpha=0.07, lw=0, transform=PC, zorder=2)
    ax.scatter(lon, lat, s=size * (0.3 + val), c=val, cmap=GLOW, vmin=0, vmax=1,
               alpha=alpha, lw=0, transform=PC, zorder=3)


def region_points(d, pl):
    """True centre and mean predicted position of each region (spherical means)."""
    d = d.merge(pl[["pid", "region", "lat", "lon"]], on="pid")
    out = {}
    for reg, g in d.groupby("region"):
        t = geo.to_xyz(g.lat.values, g.lon.values).mean(0)
        p = geo.to_xyz(g.pred_lat.values, g.pred_lon.values).mean(0)
        out[reg] = (geo.to_latlon(t), geo.to_latlon(p))
    return out


def arrows(ax, pts, labels=True, fontsize=8.5):
    for reg, lab in ARROWS.items():
        if reg not in pts:
            continue
        (tla, tlo), (pla, plo) = pts[reg]
        ax.plot([tlo, plo], [tla, pla], color=ICE, lw=1.0, alpha=0.75, transform=ccrs.Geodetic(),
                zorder=5)
        ax.scatter([plo], [pla], s=16, color=ICE, edgecolor=BG, lw=0.6, transform=PC, zorder=6)
        ax.scatter([tlo], [tla], s=30, facecolor="none", edgecolor=TEXT, lw=0.9, transform=PC,
                   zorder=6)
        if labels:
            ax.text(tlo, tla, " " + lab, fontsize=fontsize, color=TEXT, transform=PC, zorder=7,
                    va="bottom", ha="left", path_effects=HALO)


def map_panel(ax, long, pl, ct, s, curves, arrow_labels=True):
    d = long[long.step == s]
    glow_scatter(ax, d.pred_lon, d.pred_lat, count_brightness(ct, d.pid.values), size=1.4, alpha=0.75)
    arrows(ax, region_points(d, pl), labels=arrow_labels)
    wk = curves[(curves.step == s) & (curves.region == "World")]
    ax.text(0.0, 1.03, step_label(s), transform=ax.transAxes, fontsize=13, color=TEXT, weight="bold")
    if len(wk):
        ax.text(1.0, 1.03, "{}  ·  median error {:,.0f} km".format(tokens_label(s), wk.median_km.iloc[0]),
                transform=ax.transAxes, fontsize=9.5, color=SUBTLE, ha="right")


def hero(long, pl, ct, curves, size):
    lnd = land()
    steps = [s for s in HERO_STEPS if s in set(long.step)]
    fig = plt.figure(figsize=(16, 14.6))
    fig.text(0.04, 0.962, "Watching a Model Learn the World", fontsize=30, weight="bold")
    fig.text(0.04, 0.937, "Where a linear probe on Pythia-{}'s hidden states puts 15,000 places, from "
             "initialisation to the end of training".format(size), fontsize=13, color=SUBTLE)
    w, h = 0.45, 0.245
    for i, s in enumerate(steps):
        r, c = divmod(i, 2)
        ax = world_ax(fig, [0.04 + c * 0.475, 0.645 - r * 0.282, w, h], lnd)
        map_panel(ax, long, pl, ct, s, curves, arrow_labels=(i == len(steps) - 1 or i == 0))
    lx = 0.04
    fig.text(lx, 0.05, "○  where a region's places really are (their centre)", fontsize=10, color=TEXT)
    fig.text(lx, 0.031, "●  where the model puts them on average; the line joins the two",
             fontsize=10, color=ICE)
    fig.text(0.515, 0.05, "Dots: every place at its predicted position, brighter if the Pile "
             "names it more often.", fontsize=10, color=SUBTLE)
    fig.text(0.515, 0.031, "Out-of-fold ridge probes on the layer chosen in training folds.",
             fontsize=10, color=SUBTLE)
    fig.text(0.04, 0.01, "Places: Gurnee & Tegmark (2023), names cut before any comma or bracket. "
             "Counts: infini-gram index of the Pile. Coastlines: Natural Earth.", fontsize=8.5, color=FAINT)
    fig.savefig(os.path.join(C.FIGURES, "fig1_hero.png"), dpi=150)
    plt.close(fig)


def known(long, pl, size):
    lnd = land()
    steps = [s for s in HERO_STEPS if s in set(long.step)]
    fig = plt.figure(figsize=(16, 14.6))
    fig.text(0.04, 0.962, "Where it knows", fontsize=28, weight="bold")
    fig.text(0.04, 0.937, "Every place at its true position, lit by how close Pythia-{}'s probe "
             "puts it".format(size), fontsize=13, color=SUBTLE)
    w, h = 0.45, 0.245
    for i, s in enumerate(steps):
        r, c = divmod(i, 2)
        ax = world_ax(fig, [0.04 + c * 0.475, 0.645 - r * 0.282, w, h], lnd)
        d = long[long.step == s].merge(pl[["pid", "lat", "lon"]], on="pid")
        glow_scatter(ax, d.lon, d.lat, C.brightness(d.err_km.values))
        ax.text(0.0, 1.03, step_label(s), transform=ax.transAxes, fontsize=13, weight="bold")
        ax.text(1.0, 1.03, "within 500 km: {:.0%}".format((d.err_km < 500).mean()),
                transform=ax.transAxes, fontsize=9.5, color=SUBTLE, ha="right")
    cax = fig.add_axes([0.04, 0.045, 0.25, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=GLOW, norm=plt.Normalize(0, 1)), cax=cax,
                      orientation="horizontal")
    cb.outline.set_visible(False)
    ticks = [3000, 1000, 300, 100]
    cb.set_ticks([float(C.brightness(np.array([k]))[0]) for k in ticks])
    cb.set_ticklabels(["3,000 km\nor more", "1,000 km", "300 km", "100 km\nor less"])
    cax.tick_params(labelsize=9, colors=SUBTLE, length=0)
    fig.text(0.34, 0.045, "At initialisation Europe already glows: the probe's fallback guess, the "
             "middle of the sample, lies near it,\nand the probe learns some spelling from its "
             "training places. What lights up later is what the model learned.",
             fontsize=9.5, color=SUBTLE, va="center")
    fig.savefig(os.path.join(C.FIGURES, "fig2_known.png"), dpi=150)
    plt.close(fig)


def regions(curves, bs, size):
    """Per region: the model's median error over training against the spelling floor."""
    fin = bs[bs.step == bs.step.max()].set_index("region")
    order = [r for r in fin.sort_values("gain", ascending=False).index if r != "World"]
    n, cols = len(order), 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(15, 2.9 * rows + 1.4), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.06, right=0.98, hspace=0.62, wspace=0.3)
    fig.text(0.06, 0.95, "First the letters, then the world", fontsize=22, weight="bold")
    fig.text(0.06, 0.915, "Pythia-{}: median error in each region over training (amber) against what "
             "the spelling of the names alone gives (dashed).\nSorted by what training adds "
             "beyond spelling by the end.".format(size), fontsize=11.5, color=SUBTLE, va="top")
    for ax, reg in zip(axs.flat, order):
        g = curves[curves.region == reg].sort_values("step")
        sp = fin.loc[reg, "spell_km"]
        ax.plot(xstep(g.step), g.median_km, color=AMBER, lw=2)
        ax.axhline(sp, color=ICE, lw=1.1, ls="--")
        log_step_axis(ax, g.step)
        lo, hi = min(g.median_km.min(), sp), max(g.median_km.max(), sp)
        pad = 0.12 * (hi - lo + 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
        ax.set_title(ARROWS.get(reg, reg), fontsize=11.5, color=TEXT, loc="left")
        ax.text(1.0, 1.04, "beyond spelling {:+.0%}".format(fin.loc[reg, "gain"]), transform=ax.transAxes,
                fontsize=9, color=AMBER if fin.loc[reg, "gain"] > 0.03 else SUBTLE, ha="right", va="bottom")
        ax.grid(axis="y", color=LACE, lw=0.6)
        ax.tick_params(labelsize=8.5)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].set_ylabel("median error, km")
    fig.savefig(os.path.join(C.FIGURES, "fig3_regions.png"), dpi=150)
    plt.close(fig)


def beyond_fig(bss):
    """Gain over spelling through training, one panel per model: the world in amber,
    every region in grey, labelled at the right end."""
    sizes = list(bss)
    fig, axs = plt.subplots(1, len(sizes), figsize=(8.2 * len(sizes), 7), sharey=True, squeeze=False)
    axs = axs[0]
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.06, right=0.9, wspace=0.42)
    fig.text(0.06, 0.93, "What the model knows that the letters don't say", fontsize=20, weight="bold")
    fig.text(0.06, 0.885, "The model's probe pooled with the spelling baseline, against the same pool on "
             "the untrained network (out of fold). Every region starts at zero.", fontsize=11, color=SUBTLE)
    for ax, size in zip(axs, sizes):
        bs = bss[size]
        ends = []
        for reg, g in bs.groupby("region"):
            if reg == "World":
                continue
            g = g.sort_values("step")
            ax.plot(xstep(g.step), 100 * g.gain, color="#4a525e", lw=1.2)
            ends.append([100 * g.gain.iloc[-1], ARROWS.get(reg, reg)])
        w = bs[bs.region == "World"].sort_values("step")
        ax.plot(xstep(w.step), 100 * w.gain, color=AMBER, lw=3)
        ends.append([100 * w.gain.iloc[-1], "World"])
        ax.axhline(0, color=SUBTLE, lw=0.8)
        log_step_axis(ax, w.step)
        ax.set_xlim(40, 1.6e5)
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 1.9)
        for (y0, lab), y in zip(ends, ys):
            ax.annotate(lab, (1.43e5, y0), xytext=(1.8e5, y), textcoords="data", fontsize=9,
                        color=AMBER if lab == "World" else TEXT, va="center",
                        weight="bold" if lab == "World" else "normal",
                        arrowprops=dict(arrowstyle="-", color=LACE, lw=0.6), annotation_clip=False)
        ax.set_title("Pythia-{}".format(size), fontsize=13, color=TEXT, loc="left")
        ax.set_xlabel("training step")
        ax.grid(color=LACE, lw=0.6)
    axs[0].set_ylabel("error cut since initialisation, beyond spelling, %")
    fig.savefig(os.path.join(C.FIGURES, "fig4_beyond.png"), dpi=150)
    plt.close(fig)


def counts_fig(d, cms, size):
    """Left: within-country error against co-mentions (binned partial residuals), for
    `size`. Right: the coefficients for each model size, spelling held fixed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.07, right=0.97, wspace=0.42)
    fig.text(0.07, 0.93, "Named next to its country, a place is placed better", fontsize=20, weight="bold")
    fig.text(0.07, 0.885, "Final checkpoints, compared within each country (country fixed effects). Held fixed: "
             "the name's raw count, the spelling baseline's error,\npage views, place type, distance "
             "from the middle of the sample and name length.", fontsize=10.5, color=SUBTLE, va="top")
    ctrl = "C(iso) + C(etype) + log_views + log_null + ntok_llama + log_count + log_spell"
    dd = d.dropna(subset=["log_own", "log_count", "log_err", "ntok_llama", "log_spell"])
    ry = A.resid("log_err ~ " + ctrl, dd)
    rx = A.resid("log_own ~ " + ctrl, dd)
    q = pd.qcut(rx, 20, labels=False, duplicates="drop")
    gx, gy = rx.groupby(q).mean(), ry.groupby(q).mean()
    a1.scatter(gx, 100 * (10 ** gy - 1), s=42, color=AMBER, zorder=3)
    b = np.polyfit(rx, ry, 1)[0]
    xs = np.linspace(rx.quantile(0.02), rx.quantile(0.98), 50)
    a1.plot(xs, 100 * (10 ** (b * xs) - 1), color=ICE, lw=1.5)
    a1.axhline(0, color=SUBTLE, lw=0.8)
    a1.set_xlabel("co-mentions with its own country, log10, relative to similar places")
    a1.set_ylabel("error relative to similar places, %")
    a1.set_title("Pythia-{}: 20 bins of places".format(size), fontsize=10.5, color=SUBTLE, loc="left")
    a1.grid(color=LACE, lw=0.6)
    rows = [("own|count,spell", "named next to its own country"),
            ("count|own,spell", "named at all (raw string count)"),
            ("placebo|own", "named next to another country")]
    sizes = list(cms)
    for i, (k, lab) in enumerate(rows):
        for j, sz in enumerate(sizes):
            cm = cms[sz]
            if k not in cm.index:
                continue
            r = cm.loc[k]
            y = i + (j - (len(sizes) - 1) / 2) * 0.22
            col = BINS[[1, 3, 4][j]] if "placebo" not in k else FAINT
            a2.plot([r.lo, r.hi], [y, y], color=col, lw=3, solid_capstyle="round")
            a2.scatter([r.beta], [y], s=40, color=col, zorder=3)
            if i == 0:
                a2.text(r.hi + 0.004, y, sz, color=TEXT, fontsize=9, va="center")
    a2.axvline(0, color=SUBTLE, lw=0.8)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([lab for _, lab in rows], fontsize=10.5)
    a2.invert_yaxis()
    a2.set_xlabel("change in log10 error per tenfold more mentions")
    a2.set_title("each held fixed for the others; 95% intervals by country", fontsize=10.5,
                 color=SUBTLE, loc="left")
    a2.grid(axis="x", color=LACE, lw=0.6)
    fig.savefig(os.path.join(C.FIGURES, "fig5_counts.png"), dpi=150)
    plt.close(fig)


def exposure_fig(cb, edges, alpha, alpha_ci, size, what="co-mentions with its country"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.07, right=0.97, wspace=0.08)
    fig.text(0.07, 0.93, "Does a place join the map after a fixed number of mentions?",
             fontsize=20, weight="bold")
    fig.text(0.07, 0.885, "Pythia-{}: median error of places grouped by {} in the Pile. If mentions "
             "seen so far were all that mattered,\nthe right panel's curves would lie on top of "
             "each other.".format(size, what), fontsize=11, color=SUBTLE, va="top")
    for b, g in cb.groupby("bin"):
        g = g.sort_values("step")
        e = edges.loc[b]
        lab = "never" if b == 0 else "{:,.0f} to {:,.0f}".format(e["min"], e["max"])
        col = BINS[int(b) % len(BINS)]
        a1.plot(xstep(g.step), g.median_km, color=col, lw=2, label=lab)
        gg = g[(g.step > 0) & (g.exposures > 0)]
        if len(gg):
            a2.plot(gg.exposures, gg.median_km, color=col, lw=2)
    log_step_axis(a1, cb.step)
    a1.set_xlabel("training step")
    a1.set_ylabel("median error, km")
    a2.set_xscale("log")
    a2.set_xlabel("mentions seen so far (mentions in the Pile × share of training done)")
    a1.legend(frameon=False, fontsize=9.5, labelcolor=TEXT, loc="lower left", title=what,
              title_fontproperties=dict(size=9.5)).get_title().set_color(SUBTLE)
    a2.text(0.98, 0.95, "best fit: error depends on\n{:.2f} × log mentions + log step\n"
            "[{:.2f}, {:.2f}]; 1 would mean exposure\nis all that matters".format(alpha, *alpha_ci),
            transform=a2.transAxes, ha="right", va="top", fontsize=10, color=ICE)
    for ax in (a1, a2):
        ax.grid(color=LACE, lw=0.6)
    fig.savefig(os.path.join(C.FIGURES, "fig6_exposure.png"), dpi=150)
    plt.close(fig)


def scaling_fig(sc):
    w = sc[sc.region == "World"].set_index("model")
    sizes = [s for s in C.SIZES if s in w.index]
    params = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1b": 1.0e9, "1.4b": 1.4e9, "2.8b": 2.8e9}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.11, right=0.95)
    fig.text(0.11, 0.92, "Bigger models, sharper maps", fontsize=18, weight="bold")
    fig.text(0.11, 0.87, "Median error at the final checkpoint; dashed: character n-grams of the name alone",
             fontsize=10.5, color=SUBTLE)
    x = [params[s] for s in sizes]
    ax.plot(x, w.loc[sizes, "median_km"], color=AMBER, lw=2, marker="o")
    for xi, s in zip(x, sizes):
        ax.text(xi, w.loc[s, "median_km"] * 1.02, s, ha="center", va="bottom", fontsize=9, color=TEXT)
    if "spelling" in w.index:
        ax.axhline(w.loc["spelling", "median_km"], color=ICE, lw=1.2, ls="--")
        ax.text(x[0], w.loc["spelling", "median_km"] * 0.985, "spelling", color=ICE, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("median error, km")
    ax.grid(color=LACE, lw=0.6)
    fig.savefig(os.path.join(C.FIGURES, "fig7_scaling.png"), dpi=150)
    plt.close(fig)


def gif(long, pl, ct, curves, size, frames_per_step=5):
    from io import BytesIO

    from PIL import Image
    lnd = land()
    steps = sorted(long.step.unique())
    pos = {}
    for s in steps:
        d = long[long.step == s].set_index("pid").loc[pl.pid]
        pos[s] = geo.to_xyz(d.pred_lat.values, d.pred_lon.values)
    bright = count_brightness(ct, pl.pid.values)
    world = curves[curves.region == "World"].set_index("step")
    images = []
    seq = [(steps[i], steps[i + 1], f / frames_per_step) for i in range(len(steps) - 1)
           for f in range(frames_per_step)] + [(steps[-1], steps[-1], 0.0)] * 8
    for a, b, f in seq:
        xyz = (1 - f) * pos[a] + f * pos[b]
        lat, lon = geo.to_latlon(xyz)
        fig = plt.figure(figsize=(9, 5.2), dpi=100)
        ax = world_ax(fig, [0.02, 0.02, 0.96, 0.82], lnd)
        glow_scatter(ax, lon, lat, bright, size=0.9, alpha=0.75)
        d = pd.DataFrame(dict(pid=pl.pid.values, pred_lat=lat, pred_lon=lon))
        arrows(ax, region_points(d, pl), labels=True, fontsize=6.5)
        s = a if f < 0.5 else b
        fig.text(0.03, 0.93, "Watching Pythia-{} learn the world".format(size), fontsize=14, weight="bold")
        fig.text(0.03, 0.875, "{}  ·  {}  ·  median error {:,.0f} km".format(
            step_label(s), tokens_label(s), world.loc[s, "median_km"]), fontsize=10, color=SUBTLE)
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    hold = [120] * (len(images) - 8) + [250] * 8
    images[0].save(os.path.join(C.FIGURES, "learning.gif"), save_all=True, append_images=images[1:],
                   duration=hold, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="hero,known,regions,beyond,counts,exposure,scaling,gif")
    ap.add_argument("--size", default=C.TRACK_MODEL)
    ap.add_argument("--exposure", default="own", help="which count the exposure figure uses")
    args = ap.parse_args()
    os.makedirs(C.FIGURES, exist_ok=True)
    only = args.only.split(",")
    pl = A.places()
    steps = C.TRACK_STEPS if args.size == C.TRACK_MODEL else C.BIG_STEPS
    long = A.tracked(args.size, steps)
    long = long[long.pid.isin(pl.pid)]
    curves = A.region_curves(long, pl)
    ct = A.counts()
    ready = ct is not None and ct["count"].notna().mean() > 0.95
    if "hero" in only:
        hero(long, pl, ct if ready else None, curves, args.size)
    if "known" in only:
        known(long, pl, args.size)
    if "regions" in only:
        regions(curves, A.beyond_spelling(long, pl), args.size)
    if "beyond" in only:
        bss = {}
        for sz, st in [(C.TRACK_MODEL, C.TRACK_STEPS), (C.BIG_MODEL, C.BIG_STEPS)]:
            lg = A.tracked(sz, st)
            bss[sz] = A.beyond_spelling(lg[lg.pid.isin(pl.pid)], pl)
        beyond_fig(bss)
    if "scaling" in only:
        scaling_fig(A.scaling(pl))
    if ready and ("counts" in only or "exposure" in only):
        import json
        summ = json.load(open(os.path.join(C.RESULTS, "summary.json"))).get(args.size, {})
        final = long[long.step == C.FINAL_STEP].drop(columns="step")
        d = A.place_table(final, pl, ct)
        cms = {}
        for sz in [C.TRACK_MODEL, C.BIG_MODEL, "2.8b"]:
            cmp = os.path.join(C.RESULTS, "count_models_{}.csv".format(sz))
            if os.path.exists(cmp):
                cms[sz] = pd.read_csv(cmp, index_col=0)
        if "counts" in only and cms:
            counts_fig(d, cms, args.size)
        if "exposure" in only and "alpha" in summ:
            col = args.exposure
            cb, edges = A.count_bins(long, ct.assign(count=ct[col]))
            s = summ["alpha"][col]
            exposure_fig(cb, edges, s["alpha"], (s["lo"], s["hi"]), args.size)
    if "gif" in only:
        gif(long, pl, ct if ready else None, curves, args.size)


if __name__ == "__main__":
    main()
