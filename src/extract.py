"""Hidden states for every place, from one Pythia size at one training step.

Each place is read as "Where is <name>" and the state at the name's last token is kept
(in a causal model the only position that has seen the whole name). The final checkpoint
keeps every layer; other checkpoints keep the layers in config.LAYER_FRACS.

  python extract.py --size 410m --step 8000
  python extract.py --size 1b --step 143000 --prompt title     # full titles, all places

Writes data/states/<tag>.npz: last (places, kept layers, width) float16, layers (which
ones), ntok, pid. With --purge the checkpoint is deleted from the Hugging Face cache
afterwards (never the final one), since fp32 checkpoints run to 5.6 GB each.
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import config as C


def kept_layers(n_layers, step):
    """Indices into hidden_states (0 = embeddings, n_layers = last block)."""
    if step == C.FINAL_STEP:
        return list(range(n_layers + 1))
    return sorted({int(round(f * n_layers)) for f in C.LAYER_FRACS})


def encode(tok, names, template=C.TEMPLATE):
    prefix, suffix = template.split("{name}")
    assert suffix == ""
    # a bare name starts after the end-of-text token, Pythia's stand-in for a start token
    head = (tok(prefix.rstrip(" "), add_special_tokens=True).input_ids if prefix
            else [tok.eos_token_id])
    seqs = [head + tok(" " + n, add_special_tokens=False).input_ids for n in names]
    return seqs, len(head)


@torch.inference_mode()
def run(size, step, names, batch=64, threads=4, template=C.TEMPLATE):
    torch.set_num_threads(threads)
    rev = C.revision(step)
    tok = AutoTokenizer.from_pretrained(C.hub(size), revision=rev)
    net = AutoModel.from_pretrained(C.hub(size), revision=rev, dtype=torch.float32)
    net.eval()
    layers = kept_layers(net.config.num_hidden_layers, step)
    seqs, nhead = encode(tok, names, template)
    n, d = len(names), net.config.hidden_size
    out = np.zeros((n, len(layers), d), np.float16)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    order = np.argsort([len(s) for s in seqs])
    t0 = time.time()
    for b in range(0, n, batch):
        idx = order[b:b + batch]
        width = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            att[j, :len(seqs[i])] = 1
        hs = net(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        hs = torch.stack([hs[l] for l in layers], 1)      # (batch, kept, T, d)
        ends = torch.tensor([len(seqs[i]) - 1 for i in idx])
        out[idx] = hs[torch.arange(len(idx)), :, ends].numpy()
        if (b // batch) % 50 == 0:
            done = b + len(idx)
            print("  {:,}/{:,}  {:.0f} places/s".format(done, n, done / (time.time() - t0)),
                  flush=True)
    ntok = np.array([len(s) - nhead for s in seqs], np.int16)
    return out, np.array(layers, np.int16), ntok, net.config.num_hidden_layers


def purge(size, step):
    """Delete one checkpoint's files from the Hugging Face cache."""
    if step == C.FINAL_STEP:
        return
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id != C.hub(size):
            continue
        hashes = [r.commit_hash for r in repo.revisions if C.revision(step) in r.refs]
        if hashes:
            strategy = info.delete_revisions(*hashes)
            print("purging {} ({:.1f} GB)".format(C.revision(step), strategy.expected_freed_size / 1e9))
            strategy.execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=C.SIZES)
    ap.add_argument("--step", type=int, default=C.FINAL_STEP)
    ap.add_argument("--prompt", default="name", choices=["name", "title", "bare"],
                    help="name: the template on cut names; title: full titles, every place "
                         "(replication); bare: cut names with no template (robustness)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ATLAS_THREADS", 8)))
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(C.STATES, exist_ok=True)
    tag = C.tag(args.size, args.step, args.prompt)
    path = os.path.join(C.STATES, tag + ".npz")
    if os.path.exists(path) and not args.force:
        print("exists:", path)
        return
    template = "{name}" if args.prompt == "bare" else C.TEMPLATE
    if args.prompt in ("name", "bare"):
        places = pd.read_parquet(os.path.join(C.DATA, "places.parquet"))
        names = places.name.tolist()
    else:
        places = pd.read_parquet(os.path.join(C.DATA, "places_all.parquet"))
        names = places.title.tolist()
    t0 = time.time()
    last, layers, ntok, nl = run(args.size, args.step, names, args.batch, args.threads, template)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, last=last, layers=layers, ntok=ntok, pid=places.pid.values, n_layers=nl)
    os.replace(tmp, path)
    print("{}: {:,} places, layers {}, width {}, {:.1f} min".format(
        tag, last.shape[0], layers.tolist(), last.shape[2], (time.time() - t0) / 60), flush=True)
    if args.purge:
        purge(args.size, args.step)


if __name__ == "__main__":
    main()
