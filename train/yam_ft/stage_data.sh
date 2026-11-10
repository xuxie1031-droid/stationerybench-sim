#!/bin/bash
# Copy the datasets of a YAM_FT_REPOS mixture to each host's local disk (shared network FS -> local SSD), so
# training decodes videos from local disk.  Then run with DATA_ROOT=$GYRE_SCRATCH/lerobot_data (STAGE_DST).
# Also writes the wrapper's .lerobot_download_complete marker in every staged dataset: non-zero ranks otherwise wait
# for a marker that rank 0 writes on ITS local disk only (-> NCCL timeout at dataloader build, seen 11-04 13:20).
#   HOSTS="local host2 ..." train/yam_ft/stage_data.sh "<mixture string>"      (rsync, incremental; ~50-70 GB per host)
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
MIX=$1; SRC=$LEROBOT_DATA_ROOT; DST=${STAGE_DST:-$GYRE_SCRATCH/lerobot_data}
read -ra HS <<< "${HOSTS:?set HOSTS}"
LIST=$(python3 - "$MIX" "$SRC" <<'PY'
import sys, glob, os
root = sys.argv[2]; out = []
for item in sys.argv[1].split(","):
    pat = item.strip().rpartition(":")[0] or item.strip()
    for p in glob.glob(os.path.join(root, pat, "meta", "info.json")):
        rel = os.path.relpath(os.path.dirname(os.path.dirname(p)), root)
        if rel.startswith(("sim/", "hold/")) and not os.path.exists(os.path.join(os.path.dirname(p), "yam_ft_source.json")):
            continue  # conversion still in progress
        out.append(rel)
print(" ".join(sorted(set(out))))
PY
)
echo "$(echo $LIST | wc -w) datasets"
on(){ h=$1; shift; if [ "$h" = local ]; then bash -c "$*"; else ssh $h "$*"; fi; }
for h in "${HS[@]}"; do
  on $h "mkdir -p $DST; cd $SRC; t=\$(date +%s); printf '%s\n' $LIST | xargs -P 8 -I{} sh -c 'mkdir -p $DST/\$(dirname {}); nice -n 19 ionice -c3 rsync -a $SRC/{}/ $DST/{}/ && touch $DST/{}/.lerobot_download_complete'; echo $h staged \$(du -sh $DST | cut -f1) in \$((\$(date +%s)-t)) s" &
done
wait
