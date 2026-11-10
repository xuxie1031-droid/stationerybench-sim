# Reproducing the results

## 1. Environment (once)

See [environment.md](../environment.md): MolmoAct2 at commit `6070080` plus our one-line patch, the sim/server venv
(`uv sync`), the YAM robot assets, and (for training only) the training venv.

```bash
source scripts/env.sh          # all outputs go under $GYRE_WORK (default ./work); see gyre_paths.py for every variable
```

## 2. Weights

```bash
scripts/download.sh base finetuned
#   $GYRE_WORK/models/MolmoAct2-BimanualYAM                 allenai/MolmoAct2-BimanualYAM (base)
#   $GYRE_WORK/models/MolmoAct2-YAM-StationeryBench-Sim     xuxie1031/MolmoAct2-YAM-StationeryBench-Sim (fine-tuned)
```
The fine-tuned weights are the HF export of the final round's merged EMA checkpoint (see [TRAINING.md](TRAINING.md)).
They must be served with `eval/host_server_yam_noclip.py` (scripts/eval.sh does this for any HF-format directory);
the base model is served with MolmoAct2's stock server (`--model base`).

## 3. Evaluate

```bash
# smoke reproduction: 10 sealed seeds per task, nominal2, full length (about 0.5-1.5 h on one L40S)
scripts/repro_check.sh $GYRE_WORK/models/MolmoAct2-YAM-StationeryBench-Sim
# the release numbers: 50 sealed seeds per task
scripts/eval.sh all nominal2 3000000 50 --model $GYRE_WORK/models/MolmoAct2-YAM-StationeryBench-Sim --arm release_ft --confirm
scripts/eval.sh all nominal2 3000000 50 --model base --arm release_base --confirm
"$SIM_PY" eval/report.py release_ft --visual nominal2 --seeds sealed --ref release_base
```

What a run does: `scripts/eval.sh` starts the policy server on `127.0.0.1:8230`, waits for it, and calls
`eval/driver.py`, which runs chunks of up to 10 episodes, one simulator process at a time, through
`eval/run_eval_visual.py --visual nominal2 --runner eval/run_eval_<task>.py` (MolmoAct2's `sim_eval.run_eval` with our env
registered). Each episode executes full action chunks at 30 Hz until success or the step limit (eraser 3600, ruler
2000). Per-episode rubric stages are parsed from the runner output; chunks that stall are killed and their missing
seeds re-run. Results: `$GYRE_EVAL_DIR/results.json` and `$GYRE_EVAL_DIR/runs/<arm>/<visual>/<task>/`.

Seed sets (env seeds; they fix the object layout and the visual look of each episode):

| set | seeds | use |
|---|---|---|
| dev | 1000-1019 | development / milestone gate during the project |
| sealed | 3000000-3000049 | release numbers only; never used for development or tuning (`--confirm` required) |

**Sampling is unseeded.** The policy's flow-matching action sampler draws fresh noise at every call, so two runs of the
same checkpoint on the same seed can differ. Expect your counts to scatter around ours; compare with the Wilson 95%
intervals that `eval/report.py` prints (at 50 episodes and a 30% rate, the interval is roughly +-13 points).

Sanity checks on a run:
* every eraser chunk's `eval.log` contains `[eraser_task] LID_SPRING=1` (scripts/eval.sh prints the count);
* `missing []` at the end of each task in the driver output.

### Expected numbers (sealed set, nominal2)

| model | eraser_from_box | ruler_handover |
|---|---|---|
| base | 0 / 50 (0%, 95% CI 0–7%) | 0 / 50 (0%, 95% CI 0–7%) |
| fine-tuned | 18 / 50 (36%, 95% CI 24–50%) | 38 / 50 (76%, 95% CI 63–86%) |

## 4. Re-run the fine-tuning (optional)

See [TRAINING.md](TRAINING.md) for the per-round settings. Every round is one call of
`train/yam_ft/ft2_plan.sh ft3_ae|ft3_lora` with `FT3_*` overrides, on 1-8 single-GPU nodes (global batch 16).
Exporting a checkpoint for evaluation: `train/yam_ft/export_ckpt.sh runs/<run>/ema/step<N>-merged <out>-hf`
(`NOCLIP=1`), then `scripts/eval.sh ... --model <out>-hf`.

Caveat 1: the mixture each round actually sampled differs from the configured one (datasets with a configured share
below 0.1 % are over-sampled by the loader; see [TRAINING.md](TRAINING.md#configured-vs-effective-mixture)). The code
is kept as run, so re-running a round with the same datasets reproduces the same effective mixture; each run prints it
under `Sampling rates:`.

Caveat 2: the as-run mixtures also contained a third StationeryBench task that is not part of this release. Training on
the released eraser + ruler data alone redistributes that share over the two tasks (`mkmix.py` renormalises), so an
exact re-run of the lineage is not possible from the released data; the eval of the released weights is.
