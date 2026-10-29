"""Download checkpoints ahead of the extraction, so network and CPU overlap.

  python prefetch.py --size 410m [--steps 1000,3000]
"""
import argparse
import time

from huggingface_hub import snapshot_download

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default=C.TRACK_MODEL)
    ap.add_argument("--steps", default="")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")] if args.steps else (
        C.TRACK_STEPS if args.size == C.TRACK_MODEL else C.BIG_STEPS)
    for s in steps:
        t0 = time.time()
        path = snapshot_download(C.hub(args.size), revision=C.revision(s),
                                 allow_patterns=["*.json", "model.safetensors"])
        print("{} {}: {:.0f} s  {}".format(args.size, C.revision(s), time.time() - t0, path), flush=True)


if __name__ == "__main__":
    main()
