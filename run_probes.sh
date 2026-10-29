#!/usr/bin/env bash
# Probe every hidden-state file that has no predictions yet; loops until stopped, so it
# can run beside the extraction. Two light threads, so the machine stays usable.
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while true; do
  for f in ../data/states/*.npz; do
    t=$(basename "$f" .npz)
    case "$t" in *.tmp) continue ;; esac
    case "$t" in
      *.title) [ -f "../results/preds/$t.replicate.csv" ] || python probe.py --replicate "$t" --jobs 2 ;;
      *)       [ -f "../results/preds/$t.parquet" ] || python probe.py --tag "$t" --jobs 2 ;;
    esac
  done
  [ -f ../logs/probes.stop ] && break
  sleep 60
done
