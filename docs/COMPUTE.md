# Compute

Hardware: NVIDIA L40S 48 GB, one GPU per node, no NVLink / RDMA between nodes. GPU-hours below are
(number of GPUs) x (wall time from the first to the last line of the rank-0 training log), so they include data loading,
checkpointing and EMA overhead.

## Fine-tuning: lineage of the released checkpoint (base → ft12)

| round | trainable | GPUs | wall time (h) | GPU-hours |
|---|---|---|---|---|
| ft1 | action expert | 1 | 9.1 | 9.1 |
| ft2vis | action expert | 2 | 2.8 | 5.6 |
| ft2 stage 1 | action expert | 2 | 9.8 | 19.6 |
| ft2 stage 2 | action expert | 2 | 2.5 | 5.0 |
| ft3 | action expert | 4 | 4.1 | 16.2 |
| ft3b | action expert | 4 | 5.8 | 23.2 |
| ft4 | action expert | 4 | 5.7 | 22.8 |
| ft6 | action expert | 4 | 4.6 | 18.2 |
| ft7 | action expert | 4 | 4.6 | 18.4 |
| ft9 | action expert | 4 | 4.6 | 18.4 |
| ft10 | vision LoRA + LLM LoRA + action expert | 4 | 7.9 | 31.5 |
| ft11 | vision LoRA + LLM LoRA + action expert | 4 | 7.8 | 31.3 |
| ft12 | vision LoRA + LLM LoRA + action expert | 8 | 5.4 | 43.5 |
| **total** | | | | **~263** |

H100-equivalent: ~66–105 H100-hours at 2.5–4 L40S-hours per H100-hour (BF16; see
[FINETUNE_COMPUTE_SURVEY.md](FINETUNE_COMPUTE_SURVEY.md) for the conversion sources).

Side branches that are not in the lineage (abandoned recipes, a parallel LoRA line, aborted launches, smoke tests) used
another ~80 L40S GPU-hours; all fine-tuning in the project was ~340 L40S GPU-hours.

## Measured rates

Action-expert-only training ~0.65 s / sample, vision-LoRA + action expert ~1.26 s / sample on one L40S; a 4-node LoRA
round ran at ~6.7 s / step and an 8-node round at ~4.6 s / step (global batch 16, gradient sync once per optimizer
step). A LoRA round's checkpoint set (live + EMA, unmerged + merged) is ~80 GB.

## Synthetic data generation and evaluation

Synthetic data generation (simulation and rendering) and closed-loop evaluation ran on the same L40S nodes and are not counted in the fine-tuning figures above. Evaluating one model on 50 episodes of one task takes about 2–3 L40S GPU-hours (10-episode chunks of 25–40 minutes). Over the whole project, evaluation and data generation together used several hundred additional L40S GPU-hours.
