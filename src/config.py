"""Paths, models, checkpoints and every modelling choice that is not code."""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
STATES = os.path.join(DATA, "states")
RESULTS = os.path.join(HERE, "results")
PREDS = os.path.join(RESULTS, "preds")
FIGURES = os.path.join(HERE, "figures")
DOCS = os.path.join(HERE, "docs")

PLACES_URL = ("https://raw.githubusercontent.com/wesg52/world-models/main/"
              "data/entity_datasets/world_place.csv")

# Pythia: one data order shared by every size, 143,000 steps of 2,097,152 tokens, one
# pass over the Pile. Checkpoint branches on the Hub are step0, step1, 2, 4, ..., 512,
# then every 1,000 steps.
FINAL_STEP = 143000
TOKENS_PER_STEP = 2097152
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
# The model the map is watched in. Roughly log-spaced from step 1,000.
TRACK_MODEL = "410m"
TRACK_STEPS = [0, 128, 512, 1000, 2000, 3000, 4000, 6000, 8000, 11000, 16000, 23000,
               32000, 45000, 64000, 90000, 113000, 143000]
# A bigger model at fewer checkpoints: a sharper map for the hero, and a check that the
# order regions arrive in does not depend on size
BIG_MODEL = "1.4b"
BIG_STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 23000, 32000, 45000, 64000, 90000, 143000]


def hub(size):
    return "EleutherAI/pythia-" + size


def revision(step):
    return "main" if step == FINAL_STEP else "step{}".format(step)


def tag(size, step, prompt="name"):
    return "pythia-{}@{}{}".format(size, step, "" if prompt == "name" else "." + prompt)


# Each place is read in the template Godey et al. (2024) used; only tokens up to the
# end of the name are run, so the question mark never matters. The prefix also keeps
# the name off position 0, which acts as an attention sink in a causal model.
TEMPLATE = "Where is {name}"

# Hidden-state layers kept for checkpoints, as fractions of depth (the final checkpoint
# of each size keeps every layer). 0 is the embedding output.
LAYER_FRACS = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

# Place sample: names are cut to what precedes ", " or " (" so the title does not name
# the country ("Burgdorf, Switzerland" -> "Burgdorf"); names that then collide with
# another place are dropped; each country contributes at most CAP places so that the
# probe is not a map of the United States, Britain and India.
CAP = 250
FOLDS = 5
SEED = 0

EARTH_KM = 6371.0
# Map brightness: full for a place put within BRIGHT_KM[0], dark beyond BRIGHT_KM[1],
# log scale between. Absolute, so a region is not lit just for being near the middle.
BRIGHT_KM = (100.0, 3000.0)


def brightness(err_km):
    lo, hi = BRIGHT_KM
    return np.clip(np.log(hi / np.maximum(err_km, 1e-9)) / np.log(hi / lo), 0, 1)


# Display regions, from Natural Earth's UN subregions
REGION_OF = {
    "Northern America": "North America",
    "Central America": "Central America & Caribbean",
    "Caribbean": "Central America & Caribbean",
    "South America": "South America",
    "Northern Europe": "Northern & Western Europe",
    "Western Europe": "Northern & Western Europe",
    "Southern Europe": "Southern & Eastern Europe",
    "Eastern Europe": "Southern & Eastern Europe",
    "Western Asia": "Middle East & North Africa",
    "Northern Africa": "Middle East & North Africa",
    "Western Africa": "Sub-Saharan Africa",
    "Eastern Africa": "Sub-Saharan Africa",
    "Middle Africa": "Sub-Saharan Africa",
    "Southern Africa": "Sub-Saharan Africa",
    "Southern Asia": "South Asia",
    "Central Asia": "Central Asia",
    "Eastern Asia": "East Asia",
    "South-Eastern Asia": "Southeast Asia",
    "Australia and New Zealand": "Australia & New Zealand",
    "Melanesia": "Pacific Islands",
    "Micronesia": "Pacific Islands",
    "Polynesia": "Pacific Islands",
}

# infini-gram index of the Pile's training split (Llama-2 tokenizer)
INDEX = "v4_piletrain_llama"
WINDOW = 50    # tokens: a place "co-mentioned" with its country is within this distance
