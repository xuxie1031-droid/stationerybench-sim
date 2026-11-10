#!/usr/bin/env bash
# Quick reproduction check on one L40S (or similar 48 GB GPU):
#   scripts/repro_check.sh MODEL_DIR [ARM_NAME]
# Runs eraser_from_box and ruler_handover on the first 10 sealed seeds (3000000-3000009) in the nominal2 look, full
# length (efb 3600 steps on the spring lid, rh 2000 steps), and prints the success counts.  Roughly 0.5-1.5 GPU-hours
# (failed episodes run the full horizon, so a weak policy takes longest).
# Sampling is unseeded, so expect the counts to scatter around the released rates (see docs/RESULTS.md for the
# 50-seed numbers and their Wilson intervals); 10 seeds only tell you the pipeline works, not the exact rate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODEL=${1:?MODEL_DIR (HF-format checkpoint, or "base")}
ARM=${2:-repro_$( [ "$MODEL" = base ] && echo base || basename "$(readlink -f "$MODEL")" )}
bash "$GYRE_ROOT/scripts/eval.sh" all nominal2 3000000 10 --model "$MODEL" --arm "$ARM" --confirm | tee "$GYRE_EVAL_DIR/logs/repro_check_${ARM}.txt"
"$SIM_PY" - "$ARM" <<'PY'
import json, os, sys
arm = sys.argv[1]
R = json.load(open(os.path.join(os.environ["GYRE_EVAL_DIR"], "results.json")))
for r in R:
    if r["arm"] == arm and r["visual"] == "nominal2":
        print(f"REPRO {arm} {r['task']}: {r['success']}/{r['n_done']} successes (seeds {r['seeds'][0]}-{r['seeds'][-1]}, "
              f"{r['max_episode_steps']} steps{', spring lid' if r.get('lid_spring') else ''}); missing {r['missing']}")
PY
