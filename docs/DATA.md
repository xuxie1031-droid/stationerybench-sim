# Data

## Released training datasets

All training data for the two released tasks are synthetic demonstrations generated in our simulation replica, published on Hugging Face in LeRobot v3.0 format. `scripts/download.sh eraser ruler` fetches them into `$LEROBOT_DATA_ROOT/sim/`.

| Task | Hugging Face dataset | Unique trajectories | Rendered episodes |
| --- | --- | --- | --- |
| `eraser_from_box` | [stationerybench-sim-eraser](https://huggingface.co/datasets/xuxie1031/stationerybench-sim-eraser) | 1,320 | 3,503 |
| `ruler_handover` | [stationerybench-sim-ruler](https://huggingface.co/datasets/xuxie1031/stationerybench-sim-ruler) | 833 | 2,964 |

Each trajectory is rendered in two to four looks, so one trajectory appears as several episodes.

### Data sets inside each repository

| Task | Set | Episodes per look | Looks |
| --- | --- | --- | --- |
| `eraser_from_box` | `v4` | 260 | n2, dr0, dr1 |
| `eraser_from_box` | `rec3`, `rec4`, `rec5` | 233, 233, 137 | n2, dr0, dr1 |
| `eraser_from_box` | `rec6`, `rec7` | 212, 245 | n2, dr0 |
| `ruler_handover` | `v` | 237 | n2, dr0, dr1, dr2 |
| `ruler_handover` | `v3` | 228 | n2, dr0, dr1, dr2 |
| `ruler_handover` | `rec` (`_r1`, `_r2`) | 209, 159 | n2, dr0, dr1 |

`v4` (eraser) and `v` (ruler) are the base demonstration sets; `v3` (ruler) adds demonstrations executed with small injected action noise; the other sets are additional synthetic demonstration rounds generated during the project. All eraser data are recorded on the spring-return lid.

## Looks

- `n2`: `nominal2`, with lighting and materials aligned to the real Ai2 YAM camera frames. All released results are evaluated in this look.
- `dr0`, `dr1`, `dr2`: domain-randomised variants of lights, textures and materials (physics unchanged).

The looks of one trajectory show the identical motion.

## Format

- Robot type `bi_yam_follower`, 30 fps.
- Three cameras: `observation.images.top`, `.left`, `.right` (AV1, yuv420p, 360×640).
- `observation.state` and `action`: 14-D absolute joint positions `[L1..L6, L_grip, R1..R6, R_grip]`, grippers in [0, 1] (1 = open).
- The verbatim StationeryBench instruction is the language annotation.
- `meta/stats.json` carries the **pretrained** MolmoAct2-BimanualYAM normalisation statistics, so fine-tuning keeps the checkpoint's normalisation.
- Episodes are truncated at the first success plus 15 hold frames.
- `meta/yam_ft_phase.npz` holds per-frame rubric stages (ignored by LeRobot).
- Shards: `_p<k>` = every 4th episode starting at k; `_e<N>` = a conversion batch starting at sorted episode N.

## Real co-training data

The 16 public Ai2 bimanual YAM datasets listed in [TRAINING.md](TRAINING.md#real-data) (Apache-2.0) are referenced, not redistributed. `scripts/download.sh real` fetches them.

## Evaluation seeds

| Range | Use |
| --- | --- |
| 1000–1019 | development evaluation |
| 3000000–3000049 | sealed evaluation, release numbers only |

The sealed seeds were never used for development, tuning or data generation.
