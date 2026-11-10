# Training lineage

This page lists every fine-tuning round behind the released checkpoints. It covers the starting
weights, what was trained, the optimiser schedule, the data mixture, and the side branches that
were dropped. All numbers come from the training logs and launcher scripts. Where the launcher's
intent and the logged behaviour disagree, both are given.

- **Base model:** [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM), downloaded from `main` on 2026-11-02 (commit `8dcbed6` at that date).
- **Tasks (simulation, StationeryBench):**
  - `efb`: eraser_from_box.
  - `rh`: ruler_handover.
  - `msp` (middle_sticky_pad): the third StationeryBench task we worked on. It was co-trained in every round but is **not part of this release** (no code, no data) because it has not yet reached our target success rate.
- **Real data:** 16 public Ai2 bimanual-YAM LeRobot datasets (Apache-2.0), listed [below](#real-data).
- **Main line:** base → ft1 → ft2vis → ft2 (stage 1) → ft2 (stage 2) → ft3 → ft3b → ft4 → ft6 → ft7 → ft9 → ft10 → ft11 → ft12.
- The released checkpoint is the EMA export of ft12 with the LoRA adapters merged.

> **Read this first: the configured mixture is not what was sampled.**
> From round 3 on, the data loader sampled a very different mixture from the one the launcher
> requested. For example, every round was configured as sim:real = 1:1, but the real share actually
> sampled was 11 % in ft3 and 5 % in ft12. Within the sim data, `rh` received up to 84 % of all
> samples. Cause, scale and per-round numbers: [Configured vs. effective mixture](#configured-vs-effective-mixture).
> The lineage table shows the **configured** values, because those are the knobs that were set.

---

## Common recipe (all rounds unless noted)

| item | value |
|---|---|
| trainer | MolmoAct2 `train_lerobot.py`, through a thin wrapper that registers the sim+real mixture under the checkpoint's own `yam_dual_molmoact2` tag (same prompt tokens and camera keys as the base model) |
| observation / action | 3 cameras (top, left wrist, right wrist); 14-D absolute joint pose with grippers in [0, 1]; action horizon 30 at 30 fps |
| normalisation | pretrained q01/q99 stats, kept unchanged. Every sim and real dataset is stamped with the base checkpoint's stats. Every export was checked to have stats identical to the released base |
| target clipping (`NOCLIP`) | real: normalised targets clipped to [-1, 1], as in pretraining. sim: **not clipped**. See [Flags](#flags) |
| global batch | 16 (micro-batch 1 per GPU, gradient accumulation = 16 / #GPUs) |
| optimiser | AdamW, betas (0.9, 0.95), weight decay 0 on the trained groups, grad-norm clip 1.0, bf16 autocast |
| seed | 6198 (identical in every round) |
| EMA | decay 0.999 over the trainable weights, from ft2 (stage 1) onward. **Each round from ft2 stage 2 onward starts from the previous round's EMA weights** |
| AE rounds | action expert only (577.6 M trainable parameters). Vision encoder, connector and LLM frozen |
| LoRA rounds (ft10–ft12) | LoRA r = 64, alpha = 16, dropout 0.05. LLM targets: `att_proj`, `attn_out`, `ff_proj`, `ff_out`. Vision-backbone targets: `wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`, `patch_embedding`. The full action expert is also trained (737.4 M trainable parameters in total). Adapters are merged into the weights at export, and the next round injects fresh adapters on top of the merged weights |
| hardware | NVIDIA L40S 48 GB, one GPU per node, data-parallel replicas |

Schedules used:
- **cosine**: the trainer's default. Per-group linear warmup, then cosine decay to 0.1 × base.
- **cont**: continuation. No warmup; linear decay from the base LR to 1 % of it.
- **WSD**: warmup–stable–decay. Linear warmup from 0, constant, then linear decay to 0 over the last 20 % of steps.

---

## Lineage table (main line)

Task weights are relative weights of `efb / rh / msp` inside the sim share. Source shares are
fractions of a task's sim data. Visual shares split every source into its visual variants. `dr`
weight is spread across the available `dr_k` variants in proportion to frames. Data-source tags
are defined [below](#data-sources).

| round | init from | trainable · LR · warmup | steps (reached) | schedule | sim mix (configured): task weights efb / rh / msp · source shares · visual shares | sim:real (configured) | notes |
|---|---|---|---|---|---|---|---|
| **base** | – | – | – | – | – | – | `allenai/MolmoAct2-BimanualYAM` |
| **ft1** (`ft1_ae`) | base | AE · 5e-5 · 200 | 3000 (3000) | cosine | 1 / 1 / 1 · all tasks: `v2` raw 1.0 · as recorded (original simulator look) | 1:1 | 1 GPU. No EMA. Interrupted at step 1000 and resumed from the step-1000 checkpoint with an identical config |
| **ft2vis** (`ft2vis_ae`) | ft1 step 3000, **live** weights (ft1 had no EMA) | AE · 2e-5 · 100 | 1500 (1500) | cosine | 1 / 1 / 1 · all tasks: `v2r` 1.0 · nominal 0.4 / DR 0.6 | 1:1 | 2 GPUs. No EMA. Introduces the bench-aligned and domain-randomised re-renders. efb `dr_2` was only half converted at launch (5 of 10 shards) |
| **ft2 stage 1** (`ft2_ae`) | ft2vis step 1500, live weights | AE · 2e-5 · 500 | 5600 (5600) | cosine | 1.25 / 1.25 / 1 · all tasks: `v2r` 0.5 / `v3` (DART) 0.5 · nominal 0.4 / DR 0.6 | 1:1 | 2 GPUs. EMA on (fresh). `PHASE_W` on: grasp ×3, place ×2, others ×1, ±15 frames around stage changes ×2 (sim only). Planned as the first 80 % of a 7000-step run |
| **ft2 stage 2** (`ft2_ae_s2`) | ft2 stage 1 **EMA** step 5600 | AE · 2e-6 (stage-1 final LR) · 0 | 1400 (1400) | cont (to 1 %) | same as ft2 stage 1 | **1:2** (real 2/3) | 2 GPUs. `PHASE_W` on. The last 20 % of the 7000-step plan |
| **ft3** (`ft3_ae`) | ft2 stage 2 EMA step 1400 | AE · 1e-5 · 200 | 3500 (3500) | WSD | 1.25 / 1.25 / 1 · efb: `rec` (r1) 0.4 / `v2r` 0.4 / `v3` 0.2; rh: `rec` (r1) 0.4 / `v2r` 0.5 / `v3` 0.1 · nominal 0.6 / DR 0.4 | 1:1, switched **in-run** to real 2/3 at step 2800 (decay start) | 4 GPUs. First additional synthetic data round (`rec` r1). `PHASE_W` off from here on |
| **ft3b** (`ft3b_ae`) | ft3 EMA step 3500 | AE · 1e-5 · 200 | 5000 (5000) | WSD | same as ft3 | 1:1 (no switch) | 4 GPUs. **Real-data switch removed.** A first launch with the switch at step 4000 was stopped at step 100 and restarted without it |
| **ft4** (`ft4_ae`) | ft3b EMA step 5000 | AE · 1e-5 · 200 | 5000 (5000) | WSD | 1.75 / 1.25 / 1.25 · efb and rh: `rec` (r1 + r2) 0.4, other shares as ft3 · nominal 0.6 / DR 0.4 | 1:1 | 4 GPUs. Adds synthetic data round r2 |
| **ft6** (`ft6_ae`) | **ft4** EMA step 5000 (not ft5) | AE · 1e-5 · 200 | 4000 (4000) | WSD | 1.75 / 1.0 / 1.75 · efb: `v4` (prop strategy) 1.0; rh: as ft4 · nominal 0.6 / DR 0.4 | 1:1 | 4 GPUs. **Spring-return lid** for efb (all efb data from here on is recorded on it). All old-strategy efb data dropped |
| **ft7** (`ft7_ae`) | ft6 EMA step 4000 | AE · 1e-5 · 200 | 4000 (4000) | WSD | 2 / 1 / 2 · efb: `v4` 0.55 / `rec3` 0.45; rh: as ft4 · nominal 0.6 / DR 0.4 | 1:1 | 4 GPUs. First additional synthetic data round for the new strategy |
| **ft9** (`ft9_ae`) | ft7 EMA step 4000 | AE · 1e-5 · 200 | 4000 (4000) | WSD | 2 / 1 / 2 · efb: `v4` 0.4 / `rec3` 0.3 / `rec4` 0.3; rh: as ft4 · nominal 0.6 / DR 0.4 | 1:1 | 4 GPUs |
| **ft10** (`ft10_lora`) | ft9 EMA step 4000 | **LoRA** (vision r64 + LLM r64) + AE · VLM groups 2.5e-5, AE 1e-5 · 200 | 4000 (4000) | WSD | 2 / 1 / 2 · efb: `v4` 0.3 / `rec3` 0.2 / `rec4` 0.25 / `rec5` 0.25; rh: as ft4 · **nominal2 0.6** / DR 0.4 | 1:1 | 4 GPUs. First LoRA round. New `nominal2` look replaces `nominal` for every source |
| **ft11** (`ft11_lora`) | ft10 EMA step 4000, LoRA merged | LoRA + AE · 2.5e-5 / 1e-5 · 200 | 4000 (4000) | WSD | 2 / 1 / 2 · efb: `v4` 0.3 / `rec3` 0.1 / `rec4` 0.1 / `rec5` 0.15 / `rec6` 0.35; rh: as ft4 · nominal2 0.6 / DR 0.4 | 1:1 | 4 GPUs. Adds synthetic data round `rec6` |
| **ft12** (`ft12_lora`) | ft11 EMA step 4000, LoRA merged | LoRA + AE · 2.5e-5 / 1e-5 · 200 | 4000 (4000) | WSD | 2 / 1 / 2 · efb: `v4` 0.3 / `rec3` 0.05 / `rec4` 0.05 / `rec5` 0.1 / `rec6` 0.2 / `rec7` 0.3; rh: as ft4 · nominal2 0.6 / DR 0.4 | 1:1 | **8 GPUs** (2 micro-batches per rank; global batch unchanged). Adds synthetic data round `rec7`. **Released checkpoint:** EMA step 4000, merged |

"rh: as ft4" means `rec` (r1 + r2) 0.4 / `v2r` 0.5 / `v3` 0.1. msp used the same task-level
scheme with its own synthetic demonstration sets (same rounds as efb); it is not part of this release.

### How each round was launched

Every round from ft3 on is one call of [`train/yam_ft/ft2_plan.sh`](../train/yam_ft/ft2_plan.sh) in mode `ft3_ae`
(action expert only) or `ft3_lora` (LoRA + action expert) with `FT3_*` overrides (`FT3_INIT`, `FT3_STEPS`,
`FT3_TASKS`, `FT3_SRC_SHARE`, `FT3_VIS`/`FT3_VIS_CUSTOM`, `FT3_NO_SWITCH`), which calls `ft2.sh` -> `mkmix.py` (mixture
string) -> `ddpN.sh` (one process per node) -> `run_ft.sh` -> `train_yam_ft.py`. ft2 stage 1/2 used the `ae1` / `ae2`
modes. The final round, verbatim apart from host names: [`train/rounds/ft12_lora.sh`](../train/rounds/ft12_lora.sh).
Checkpoints are exported for evaluation and for the next round's init with
[`train/yam_ft/export_ckpt.sh`](../train/yam_ft/export_ckpt.sh) (`NOCLIP=1`, `EMA=1`; LoRA runs: the `-merged` checkpoint).

---

## Configured vs. effective mixture

**Mechanism.** The wrapper splits each mixture entry over its shards in proportion to frames. It
passes each dataset to the loader as `root_size_factor = (1000 · w)²`, where `w` is the dataset's
intended share. The loader turns that factor into a sampling rate in one of two ways:
- for factors ≥ 1, the rate is `sqrt(factor)`, which is proportional to `w`, as intended;
- for factors < 1, the rate is `sqrt(len(dataset) · factor)`.

So any shard whose intended share is below 0.1 % gets its rate multiplied by `sqrt(#frames)`,
roughly 100–300× too large.

**Scale.** Up to ft2 every shard stayed above the threshold, apart from 3 `rh` DART shards in
ft2 stage 2. From ft3 on, sources were split into more shards and the threshold was crossed
widely: 38 of 192 datasets in ft3, and 111 of 238 in ft12. The 16 real datasets always stayed
above it, so they were relatively under-sampled.

**Where to check.** The per-dataset rates the loader actually used are printed in every run's
rank-0 log under `Sampling rates:`. The in-run switch in ft3 reports the same pre-switch real
share (11.04 %) from the dataloader workers.

Effective shares of **all** training samples, computed from the logged sampling rates:

| round | real | efb | rh | msp | efb composition (effective) | rh composition (effective) |
|---|---|---|---|---|---|---|
| ft1 | 50.0 % | 16.7 % | 16.7 % | 16.7 % | as configured | as configured |
| ft2vis | 50.0 % | 16.7 % | 16.7 % | 16.7 % | as configured | as configured |
| ft2 stage 1 | 50.0 % | 17.9 % | 17.9 % | 14.3 % | as configured | as configured |
| ft2 stage 2 | 49.7 % (configured 66.7 %) | 8.9 % | 34.4 % | 7.1 % | as configured | `v3` 87 % (DR 82 %), `v2r` 13 % |
| ft3, steps 1–2799 | 11.0 % | 32.5 % | 37.3 % | 19.1 % | `v2r` 93 % (`dr_2` alone 89 %), `rec` r1 5 %, `v3` 2 %; nominal 7 % | `v3` 90 % (DR 76 %), `v2r` 5 %, `rec` 4 %; nominal 20 % |
| ft3, steps 2800–3500 | 66.7 % | 12.2 % | 14.0 % | 7.2 % | as above | as above |
| ft3b | 11.0 % | 32.5 % | 37.3 % | 19.1 % | as ft3 | as ft3 |
| ft4 | 7.8 % | 3.2 % | 74.9 % | 14.2 % | as configured (`v2r` 40, `rec` r1 18 / r2 22, `v3` 20) | `v2r` 56 %, `v3` 26 %, `rec` r2 17 % / r1 1 %; DR 94 % |
| ft6 | 9.0 % | 3.5 % | 84.0 % | 3.5 % | as configured | `v2r` 63 %, `v3` 24 %, `rec` r2 13 % / r1 1 %; DR 92 % |
| ft7 | 9.8 % | 3.9 % | 82.4 % | 3.9 % | as configured | as ft6 |
| ft9 | 9.8 % | 3.9 % | 82.4 % | 3.9 % | as configured | as ft6 |
| ft10 | 8.3 % | 3.3 % | 69.6 % | 18.8 % | as configured | as ft6 (nominal2 instead of nominal) |
| ft11 | 5.5 % | 27.1 % | 46.3 % | 21.1 % | `rec4` 56 %, `rec3` 37 %, `rec6` 3 %, `v4` 2 %, `rec5` 1 %; DR 95 % | as ft10 |
| ft12 | 4.9 % | 23.4 % | 41.3 % | 30.5 % | `rec3` 39 %, `rec4` 39 %, `rec5` 15 %, `rec7` 3 %, `v4` 3 %, `rec6` 2 %; DR 95 % | as ft10 |

"As configured" means the composition inside that task matches the lineage table to within
rounding. Only the task's share of the whole mixture changed.

To reproduce the *intended* mixture, give every dataset a share of at least 0.1 %, or pass
explicit per-dataset `sampling_rate`s instead of `root_size_factor`. To reproduce the released
weights exactly, use the logged rates.

---

## Data sources

All sim data are synthetic demonstrations generated in the StationeryBench ManiSkill replicas. Each one is converted to LeRobot v3.0 (360×640, 30 fps) with the base model's normalisation stats, and truncated at the first task success plus 15 hold frames. Dataset ids are `sim/<task>_<source><visual>[_r<round>][_p<shard>|_e<start>]`.

| Tag | What it is | efb episodes | rh episodes |
| --- | --- | --- | --- |
| `v2` | base demonstrations, original simulator look; used in ft1 only | 249 | 237 |
| `v2r` | the same `v2` demonstrations re-rendered in the `nominal`, `dr_0..2` and (from ft10) `nominal2` looks; ids `vnom`, `vdr0..2`, `vn2` | 249 per look | 237 per look |
| `v3` | demonstrations executed with small injected action noise (DART), clean commands as labels | 168 per look | 228 per look |
| `v4` | eraser demonstrations of the "prop" strategy on the spring-return lid (the left arm lifts the lid to ~92° and holds it while the right arm takes the eraser out) | 260 per look | – |
| `rec` r1, r2 | additional synthetic demonstration rounds | r1 257, r2 305 | r1 209, r2 159 |
| `rec3`–`rec7` | additional synthetic demonstration rounds for the `v4` strategy (efb and msp only); `rec6`/`rec7` exist only in nominal2 + dr_0 | r3 233, r4 233, r5 137, r6 212, r7 245 | – |

The released datasets contain the efb and rh sets used by the final round; see [DATA.md](DATA.md).

<a id="real-data"></a>
## Real data

16 Ai2 bimanual-YAM datasets (license `apache-2.0` on each dataset card), all downloaded in full.
Within the real share they are weighted by frame count, so sampling is uniform over real frames.
The revisions are the current `main` commits, unchanged since May 2026.

| HF dataset | episodes | frames | revision |
|---|---|---|---|
| [`allenai/03012026-plate-cleaning-07`](https://huggingface.co/datasets/allenai/03012026-plate-cleaning-07) | 13 | 25,608 | `53913c1` |
| [`allenai/12122025-tool-07`](https://huggingface.co/datasets/allenai/12122025-tool-07) | 10 | 22,437 | `50dba66` |
| [`allenai/13012026-scan-6`](https://huggingface.co/datasets/allenai/13012026-scan-6) | 10 | 23,966 | `4caa2c7` |
| [`allenai/14122025-cut-12`](https://huggingface.co/datasets/allenai/14122025-cut-12) | 10 | 23,264 | `5d4d1a3` |
| [`allenai/15122025-cup-09`](https://huggingface.co/datasets/allenai/15122025-cup-09) | 23 | 27,343 | `904b843` |
| [`allenai/15122025-toy-07`](https://huggingface.co/datasets/allenai/15122025-toy-07) | 10 | 23,641 | `f70a7e4` |
| [`allenai/16122025-toilet-09`](https://huggingface.co/datasets/allenai/16122025-toilet-09) | 32 | 25,291 | `a5a25e8` |
| [`allenai/17122025-clo-14`](https://huggingface.co/datasets/allenai/17122025-clo-14) | 30 | 28,660 | `0994f37` |
| [`allenai/17122025-med-06`](https://huggingface.co/datasets/allenai/17122025-med-06) | 20 | 26,327 | `eed45e0` |
| [`allenai/19012026-charging-08`](https://huggingface.co/datasets/allenai/19012026-charging-08) | 19 | 30,765 | `c18f5d8` |
| [`allenai/23012026-scoop-06`](https://huggingface.co/datasets/allenai/23012026-scoop-06) | 20 | 39,399 | `9d28e5b` |
| [`allenai/26122025-foldclo-04`](https://huggingface.co/datasets/allenai/26122025-foldclo-04) | 10 | 22,484 | `9a3a6e2` |
| [`allenai/26122025-tablebuss-13`](https://huggingface.co/datasets/allenai/26122025-tablebuss-13) | 15 | 26,168 | `6ae0ddf` |
| [`allenai/27012026-untangle-08`](https://huggingface.co/datasets/allenai/27012026-untangle-08) | 10 | 31,881 | `ae05e9c` |
| [`allenai/28112025-block-02`](https://huggingface.co/datasets/allenai/28112025-block-02) | 16 | 31,115 | `957f65a` |
| [`allenai/30112025-box-02`](https://huggingface.co/datasets/allenai/30112025-box-02) | 10 | 25,932 | `3d5b0da` |
| **total** | **258** | **434,281** (~4.0 h) | |

The same 16 datasets were used in every round.

---

## Side branches (not in the released lineage)

- **LoRA line from the base: `ft1_lora` → `ft2_lora` → `ft2_lora_s2`.** LoRA r64 on vision and LLM plus the action expert.
  - `ft1_lora`: 1500 steps, all LRs 5e-5.
  - `ft2_lora`: from the merged `ft1_lora` step 1500. 4000 steps, VLM 5e-5 / AE 2e-5, same data as ft2 stage 1.
  - `ft2_lora_s2`: from the merged *live* `ft2_lora` step 4000. 1000 steps, cont schedule, real 2/3.
  - Not continued: round 3 and later ran AE-only until ft10, which re-introduced LoRA on top of the AE line.
- **`ft5_ae`**: from ft4 EMA step 5000, 4000 steps planned, with efb switched to `v4` only. It crashed at step 1000 while writing the step-1000 checkpoint, after the shared disk filled up. It was dropped, and ft6 starts from ft4 instead.
- **`ft8_ae`**: a control run from the **base model** on ft6's data (task weights 2 / 1 / 2). 6000 steps, AE 2e-5, warmup 500, WSD. Development evaluation of its EMA weights: efb 0/20, rh 7/20 (msp 0/20). That failed the no-regression check against ft4 EMA, so the run was abandoned.
- **AE-only ft10 control**: planned (same data as ft10, old nominal look) but never trained.
- Short smoke, LR = 0 and multi-node consistency runs are not listed.

---

## Flags

| flag | meaning |
|---|---|
| `NOCLIP` (`real/*:clip, ref/*:clip, sim/*:noclip`) | Sim action and state targets are normalised with the pretrained q01/q99 stats but **not** clipped to [-1, 1]. Real data keeps the pretraining clip. Every run carries this rule. Serve the checkpoints with action **un-normalisation without the [-1, 1] clip**. The discrete state tokenizer still saturates at [-1, 1] |
| `PHASE_W` / `PHASE_TRANS` | Phase-weighted frame sampling inside sim datasets, using per-frame stage labels: grasp ×3, place ×2, others ×1, and frames within 15 of a stage change ×2. Task and real shares are unaffected. On in ft2 stage 1 and 2 (and `ft2_lora*`), off from ft3 |
| DART | Demonstrations executed with injected action noise while recording the clean commands as labels (`v3`, `v4`) |
| EMA | Exponential moving average of the trainable weights (decay 0.999), saved next to every checkpoint. For LoRA runs it is also saved merged. From ft2 stage 2 onward each round initialises from the previous round's EMA export |
| `MIX_SWITCH` | Optional in-run rescaling of the real share at a given step, keeping the proportions inside the sim and real groups. Used only in ft3: real share 2/3 from step 2800 |
| WSD / cont | Learning-rate schedules, see [Common recipe](#common-recipe-all-rounds-unless-noted) |
| `ACCUM_NOSYNC` | Multi-node runs started by the round launcher all-reduce gradients once per optimiser step rather than once per micro-batch. This changes speed only |
