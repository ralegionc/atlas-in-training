#!/usr/bin/env bash
# After run_track.sh: the bigger model's checkpoints, the template check (the final
# Pythia-410m read with bare names) and full titles for the replication of Godey et al.
# (2024). Each 1.4b checkpoint downloads while the previous one is read, and is deleted
# from the cache once read (fp32 1.4b checkpoints are 5.6 GB).
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
while ps -ef | grep -v grep | grep -q -E "run_track.sh|extract.py"; do sleep 30; done
steps=($(python -c "import config as C; print(' '.join(map(str, C.BIG_STEPS)))"))
python prefetch.py --size 1.4b --steps "${steps[0]}" > ../logs/prefetch_big.log 2>&1
for i in "${!steps[@]}"; do
  next="${steps[$((i + 1))]}"
  [ -n "$next" ] && { python prefetch.py --size 1.4b --steps "$next" >> ../logs/prefetch_big.log 2>&1 & }
  python extract.py --size 1.4b --step "${steps[$i]}" --purge 2>&1 | grep --line-buffered -v -i warning
  wait
done
python extract.py --size 410m --prompt bare 2>&1 | grep --line-buffered -v -i warning
for size in 160m 1b; do
  python extract.py --size "$size" --prompt title 2>&1 | grep --line-buffered -v -i warning
done
