#!/usr/bin/env bash
# Hidden states: the final checkpoint of every size, then every checkpoint of the
# tracked model. One extraction at a time, 8 threads (set ATLAS_THREADS to change it).
# Finished files are skipped, so this can be rerun.
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
# wait for any extraction already running
while ps -ef | grep -v grep | grep -q "extract.py"; do sleep 30; done
for size in 1.4b 2.8b 70m 160m 410m 1b; do
  python extract.py --size "$size" 2>&1 | grep --line-buffered -v -i warning
done
steps=$(python -c "import config as C; print(' '.join(map(str, C.TRACK_STEPS)))")
for s in 0 512 2000 8000 32000 $steps; do
  python extract.py --size 410m --step "$s" 2>&1 | grep --line-buffered -v -i warning
done
