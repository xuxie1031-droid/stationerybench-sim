# Fine-tuning success lift vs compute — public results (survey 2026-11-09)

Compiled for the Gyre StationeryBench-Sim pilot release (2026-11-09). ~65 public results checked, mostly against primary sources (arXiv, official READMEs, HF model cards, company blogs). Verification marks: P = primary source; P-fig = primary, read off a figure (±2 pp); P-self = company/individual self-report. "derived" = GPU count × hours multiplied by us. TP = task-progress score, not binary success.

**Key finding: only about a quarter of results report both GPU time and a success lift.** Physical Intelligence never published π0/π0.5/π*0.6 fine-tune compute; NVIDIA gives a GR00T fine-tune duration only once (Isaac Lab-Arena docs); SimpleVLA-RL, RLinf, πRL and OpenVLA-OFT give GPU type/count but not hours. Reporting our L40S GPU-hours explicitly is itself a differentiator.

## A. Supervised fine-tuning (full / LoRA / head / post-training)

| # | Method | Base | Task | Data | Before → after | Compute (as reported) | Source | Ver. |
|---|---|---|---|---|---|---|---|---|
| A1 | ACT / ALOHA (2304.13705) | ACT ~80M from scratch | bimanual, new; ALOHA real, 6 tasks | 50 demos/task | baselines 0 → ACT mean 74 (20–96) | ~5 h/task, 1×RTX 2080 Ti | arxiv.org/abs/2304.13705 | P |
| A2 | Octo (2405.12213) | Octo-Base 93M full FT 50k steps | 6 real new domains incl. bimanual ALOHA | ~100 demos/domain | vs scratch 20→72; bimanual 20→80 | ~5 h, 1×A5000 | arxiv.org/abs/2405.12213 | P |
| A3 | OpenVLA new-robot adaptation (2406.09246) | OpenVLA 7B full FT | Franka-Tabletop / DROID real | 10–150 demos/task | vs scratch 43.4→67.2; DROID 21.7→58.3 | 8×A100 × 5–15 h/task (derived 40–120 A100-h) | §5.2 | P |
| A4 | OpenVLA LoRA vs full | same | Franka subset | — | full 69.7, LoRA r32 68.2 | LoRA 1×A100 10–15 h ("8× less compute") | §5.3 | P |
| A5 | CogACT (2411.19650) | 7.6B full FT | Realman / Franka real | ~400 demos | vs fine-tuned OpenVLA 12.1→71.2; 6.8→61.4 | 16×A100 × 7.5 h, 10K steps (derived 120 A100-h) | arxiv.org/abs/2411.19650 | P |
| A6 | RDT-1B (2410.07864) | 1.2B multi-task full FT | bimanual ALOHA real | 6K+ demos, 300+ tasks; 1/5-shot | vs RDT scratch 40→68; 16→40; unseen cup 0→50 | 48×H100 ~3 days, 130K steps (derived ~3,456 H100-h, multi-task) | arxiv.org/abs/2410.07864 | P |
| A7 | TinyVLA (2409.12514) | 1.3B LoRA + diffusion head | bimanual UR5 real, 3 tasks | 100 demos/task | OpenVLA 0/0/0 → 76.7/36.7/30 | not reported | arxiv.org/abs/2409.12514 | P |
| A8 | π0, complex unseen tasks (2410.24164 Fig.13) | π0 3.3B full FT | bimanual, new; box folding, to-go box, eggs | 5–100+ h | out of the box all 0; scratch→pretrained TP 0.32→0.64, 0.46→0.70, 0.89→0.84 | not reported | arxiv.org/abs/2410.24164 | P-fig |
| A9 | π0 low-data (Fig.11) | same | bimanual towel folding, paper towel | 1/5/10 h | 1 h data, scratch→pretrained TP 0.00→0.15, 0.37→0.82 | not reported | same | P-fig |
| A10 | Gemini Robotics specialisation (2503.20020) | Gemini Robotics | bimanual, new; ALOHA 2 long-horizon dexterous | 2–5k demos/task | same arch from scratch 0% → mean 79% | not reported | arxiv.org/abs/2503.20020 | P |
| A11 | GR00T N1 (2503.14734) | N1 2B full post-train | sim DexMG (bimanual dexterous), GR-1; real GR-1 | 30/100/300 demos/task | DP scratch → N1: DexMG(30) 23.7→29.6; GR-1(30) 21.3→43.2; real 10% data 10.2→42.6 | steps only (20k–60k, batch 128/1024); pretrain ~50,000 H100-h | arxiv.org/abs/2503.14734 | P |
| A12 | GR00T N1.5 vs N1 | N1.5 3B | RoboCasa 30 demos/task; Unitree G1 | — | 17.4→47.5; G1 44.0→98.8 (vs previous model) | not reported | research.nvidia.com/labs/gear/gr00t-n1_5/ | P |
| A13 | GR00T N1.6 @ Isaac Lab-Arena (docs) | N1.6 3B | sim GR1 open microwave (Mimic data) | unspecified | after 0.88 (n=17) / 1.0 (n=19) / 0.605 (n=200) | **8×L40S ~4–8 h** (20k steps, batch 24; derived 32–64 L40S-h) or 1×Ada6000 2–3 h | isaac-sim.github.io/IsaacLab-Arena (step_4_policy_training) | P — only official source with both L40S time and success |
| A14 | GR00T N1.7 vs N1.6 | 3B | SimplerEnv, RoboCasa | — | Fractal 52.0→72.5; Bridge 56.6→62.3; RoboCasa 66.2→70.8 | steps only (20k × batch 1024, 8 GPU) | github.com/NVIDIA/Isaac-GR00T | P |
| A15 | SmolVLA (2506.01844) | 0.45B full FT | SO-100 real single-arm | 50 demos/dataset | without community pretraining 51.7 → with 78.3 (pretraining lift) | paper 200k steps; LeRobot docs "20k steps ~4 h on 1 A100" | arxiv.org/abs/2506.01844 | P |
| A16 | Cosmos Policy (2601.16163) | Cosmos-Predict2 2B | bimanual ALOHA 2 real, 4 tasks | 185 demos total | mean completion 93.6 vs π0.5 88.6 / π0 77.9 / OFT+ 62.0 / DP 33.6 | HF card: 8×H100 ~48 h, 50k steps (derived ~384 H100-h) | huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B | P |
| A17 | MolmoAct (2508.07917) | 7B LoRA r32 | bimanual real set_table / lift_tray / fold_towel | 50 demos/task | progress +22.7 pp over π0-FAST | post-train not reported (pretrain 9,728 H100-h) | arxiv.org/abs/2508.07917 | P |
| A18 | **MolmoAct2 YAM fine-tune (2605.02881)** | MolmoAct2 | bimanual YAM real, 8 tasks × 50 trials | dataset 34.5k episodes / 720 h | mean 50.1%, "15% above runner-up OpenVLA-OFT" (abs/rel unclear) | **64×H100, 100K steps, "roughly 2,300 GPU hours"** | arxiv.org/abs/2605.02881 | P |
| A19 | LingBot-VLA GM-100 (2601.18692) | — | bimanual, new; 4 platforms × 100 tasks real | 130 demos/task | post-trained success π0.5 13.0, GR00T N1.6 7.31, LingBot 16.87 | not reported | arxiv.org/abs/2601.18692 | P |
| A20 | RoboTwin 2.0 benchmark (2506.18088) | per-model single-task FT | bimanual sim, 50 tasks | 50 demos/task | Easy/Hard: π0 46.4/16.3, RDT 34.5/13.7, ACT 29.7/1.7, DP 28.0/0.6, DP3 55.2/5.0 | steps/batch only | arxiv.org/abs/2506.18088 | P |
| A21 | X-VLA (2510.10274) | 0.9B | bimanual RoboTwin 2.0 | — | Easy/Hard 70.0/39.0 | FT 60K steps batch 128; pretrain 64×A100 ~4 days | arxiv.org/abs/2510.10274 | P |
| A22 | TwinVLA (2511.05275) | 1.3B | bimanual real towel folding | 50 demos | RDT 15/15, π0 40/60, TwinVLA 45/25 | "25 H100 GPU-days" incl. pretraining | arxiv.org/abs/2511.05275 | P |
| A23 | ALOHA Unleashed (2410.13126) | diffusion 217M from scratch | bimanual ALOHA 2, 5 real tasks | 26k+ demos | shirt hanging ACT 25 → DP 70; others 25–95 | 64 TPUv5e ~265 h | arxiv.org/abs/2410.13126 | P |

## B. OFT-style recipe changes

| # | Method | Task | Before → after | Compute | Source | Ver. |
|---|---|---|---|---|---|---|
| B1 | OpenVLA-OFT (2502.19645) | LIBERO (saturated) | 76.5→97.1 (π0 already 94.2) | LoRA 50–150K steps, 8×A100/H100; hours not reported | arxiv.org/abs/2502.19645 | P |
| B2 | OpenVLA-OFT+ | bimanual ALOHA real, 20–300 demos/task | staged scores; pot: OFT+ 51.25 vs π0 42.08 / RDT 44.17 / DP 30.83 / ACT 23.75 | no hours for any method | same | P |
| B3 | π0-FAST / Knowledge Insulation | pretraining recipe | ratios only ("5× fewer GPU hours"; 1/7.5 steps) | absolute not reported | arxiv | P |
| B4 | LIBERO saturated group | LIBERO | π0 94.15 / π0-FAST 85.5 / π0.5 96.85 / MolmoAct2 97.2 / VLA-Adapter 97.3 | MolmoAct2 32×H100 ~1,150 GPU-h; others no hours | READMEs / papers | P |

## C. Co-training / synthetic data / augmentation

| # | Method | Task | Data | Before → after | Compute | Source | Ver. |
|---|---|---|---|---|---|---|---|
| C1 | Mobile ALOHA co-training (2401.02117) | bimanual mobile, new, real | 20–50 demos/task + static ALOHA | wash pan 0→80, call elevator 0→95, push chairs 0→80 | not reported | arxiv.org/abs/2401.02117 | P |
| C2 | DexMimicGen (2410.24185) | bimanual dexterous sim + real humanoid | 5–10 source → 1,000 generated/task | can sorting 0.7→97.3; drawer 0.7→76.0; real 0→90 | not reported | arxiv.org/abs/2410.24185 | P |
| C3 | MimicGen (2310.17596) | sim | 10 → 1,000 | three-piece assembly 1.3→82.0; Square 11.3→90.7 | not reported | arxiv | P |
| C4 | RoboTwin 2.0 sim-to-real | bimanual RDT real | 10 real + 1k synthetic | 17.0→41.4 | not reported | arxiv.org/abs/2506.18088 | P |
| C5 | DreamGen (2505.12705) | GR00T N1 real / RoboCasa | 10–13 real + 100–300 synthetic/task | SO-100 21→45.5; GR1 37→46; RoboCasa 17.44→23.32 | data generation "54 h on 1500 L40" (~81,000 L40-h) | arxiv.org/abs/2505.12705 | P |
| C6 | Sim-and-Real Co-Training (2503.24361) | DP real | 50/20 real + sim | 45.3→83.2 | not reported | arxiv | P |
| C7 | GR-3 (2507.15493) | real pick-place unseen objects | 10 human VR traj./object | 57.8→86.7 | not reported | arxiv | P |
| C8 | DreamZero (2602.15922) | video-only transfer | 20 min robot video | TP 38.3→55.4 | not reported | arxiv | P |

## D. DAgger / recovery / interactive imitation

| # | Method | Task | Before → after | Compute | Source | Ver. |
|---|---|---|---|---|---|---|
| D1 | RaC (2509.07953) | bimanual xArm real shirt hanging | 78.3% with ~5 h data (ALOHA Unleashed 75% with ~89 h, authors' estimate) | 8×L40S or 4×RTX 6000 Ada; hours not reported | arxiv.org/abs/2509.07953 | P |
| D2 | CR-DAgger (2506.16685) | real contact-rich | book flip 40→100; belt 15.6→96.8 (50–100 corrections) | 1×RTX 4090; hours not reported | arxiv | P |
| D3 | RoboCopilot (2503.07771) | real | 33.3→91.7 (BC same data 58.3) | not reported | arxiv | P |

## E. Online sim RL (incl. world-model and residual RL)

| # | Method | Task | Start | Before → after | Compute | Source | Ver. |
|---|---|---|---|---|---|---|---|
| E1 | SimpleVLA-RL (2509.09674) | LIBERO | 1 demo/task SFT | Long 17.3→91.7; mean 48.9→96.9 | 8×A800; hours not reported | arxiv.org/abs/2509.09674 | P |
| E2 | same | bimanual RoboTwin 2.0, 12 tasks | 1,000 demos/task SFT | 38.3→68.8; Lift Pot 10.1→64.1 | same | same | P |
| E3 | same, sim-to-real | bimanual AgileX Piper real, 4 tasks | 0 real demos | 17.5→38.5; Pick Bottle 0→14 | same | same | P |
| E4 | RLinf-VLA (2510.06710) | bimanual RoboTwin, 6 tasks | SimpleVLA SFT ckpt | 24.48→84.63; Pot 3.13→70.31 | not reported | arxiv.org/abs/2510.06710 | P |
| E5 | same | LIBERO-130 | — | 42.09→98.11 | not reported | same | P |
| E6 | πRL (2510.25889) | LIBERO / ManiSkill | π0 58-demo SFT | π0 57.6→97.6; π0.5 ManiSkill 40.1→90.9 | 8×H100; hours not reported | arxiv.org/abs/2510.25889 | P |
| E7 | RIPT-VLA (2505.17016) | LIBERO | 1 demo | QueST 3.5→97.2 (best pair) | 1 GPU / 4×A5000; hours not reported | arxiv | P |
| E8 | RL4VLA (2505.19789) | ManiSkill ID + OOD | SFT on 16k traj. | ID 78.1→93.8; unseen objects 45.3→71.4; robot pose 33.9→79.7 | **"about 42 hours on a single NVIDIA A100"** | arxiv.org/abs/2505.19789 | P |
| E9 | VLA-RL (2505.18719) | LIBERO (saturated) | — | 76.5→81.0 | "48 GPU hours" (type unstated) | arxiv | P |
| E10 | World-Env (2509.24948) | LIBERO | 5 demos/task | 74.85→79.6 | 8×H20 ~48 h (~384 GPU-h) | arxiv | P |
| E11 | WMPO (2511.09515) | MimicGen sim; bimanual Mobile ALOHA real | 128 / 1,280 rollouts | sim 33.6→57.6; real 53→70 | 8 / 32 H100; hours not reported | arxiv | P |
| E12 | ResiP residual PPO (2407.16677) | sim assembly | 50 demos, ≤500M env steps | round table 12→94; lamp 7→97; peg 5→99 | not reported | arxiv | P |
| E13 | Moonlake (blog 2026-10) | bimanual YAM + π0.5, bin picking | 50 teleop demos | RFT +11 progress; sim RL +38 sim / +39 real progress; grasp 16%→48%; sim-real r=0.93 over 15 policies | not reported; ~20 h sim per iteration vs ~5.5 days real | moonlakeai.com/blog/simulation-enabled-scaling-paradigm | P-self |
| E14 | RL-Co (2602.12628) | real Franka + ManiSkill | 20–50 real demos/task | π0.5 real-only 26.7 → SFT co-train 45.9 → RL-Co 66.2 | not reported | arxiv | P |
| E15 | PLD (2511.00091) | LIBERO; bimanual YAM GPU insertion | — | LIBERO π0 93.4→97.2; YAM no success rate | SFT 8×L40 LoRA; RL not reported | arxiv | P |
| E16 | RialTo (2403.03949) | real→sim→real RL | 15 real demos | BC ~25 → 91 | RTX 2080/3090, "2 days and 3 hours" per task | arxiv | P |

## F. Real-world RL / offline RL / noise-space steering

| # | Method | Task | Before → after | Robot time / compute | Source | Ver. |
|---|---|---|---|---|---|---|
| F1 | HIL-SERL (2410.21845) | 13 real tasks incl. bimanual | 49.7→100 (vs HG-DAgger BC, same data) | 1–2.5 h real training/task; 1×RTX 4090 | arxiv | P |
| F2 | ConRFT (2502.05450) | Octo-small, 8 real tasks | 39.4→96.3 | 45–90 min online; RTX A6000 | arxiv | P |
| F3 | PA-RL (2412.06685) | OpenVLA 7B real | zero-shot 40→70 | 40 min online + 1 h rollouts; TPU v4 | arxiv | P |
| F4 | DSRL (2506.15799) | π0-DROID real | zero-shot 5/20→18/20 (80 episodes) | RTX 3070 | arxiv | P |
| F5 | iRe-VLA (2501.16664) | real | 0.35→0.80 | ~1 h RL/task; 1×4090 + 4×A100 | arxiv | P |
| F6 | ResFiT (2509.19301) | bimanual Vega humanoid real | 14→64; 23→64 | 15–76 min real | arxiv | P |
| F7 | GR-RL (2512.01801) | bimanual shoe lacing real | 45.7→83.3 (online RL +10.6) | not reported | arxiv | P |
| F8 | π*0.6 / Recap (2511.14759) | real espresso, laundry, boxes | espresso ~39→~92 (fig.) | not reported | arxiv / pi.website/blog/pistar06 | P-fig |

## G. Test-time methods
- G1 RoboMonkey (2506.17811): OpenVLA real OOD 35→60; SIMPLER +9; LIBERO-Long 49.8→56.5. Verifier LoRA 8×H100 (hours not reported). P
- G2 V-GPS (2410.13816): relative only, Octo-small real +55% to +100%. P

## H. Compute only, no success rate (cost references)
- Robocurve gr00t-n1.7-so101-molmoact2: action-head LoRA (1.73% params), 2,242 episodes, 1×H100 ~11.5 h, ~$70 (project ~$130); offline loss only.
- Robocurve pi0.5-yam: full FT, 5,145 episodes, 8×B200 ~17.5 h (~140 B200-h); open-loop MSE only.
- Robocurve dreamzero-yam-molmoact2: LoRA r32 + action head, 8×B200 ~78 h (~625 GPU-h); main training ~$3,900, direct $5.1–5.5K, all-in ~$6.5K; open-loop MSE indistinguishable from π0.5/GR00T at n=20.
- Community MolmoAct2-BimanualYAM-oranges: 80 demos, 5k steps, 8 GPUs × batch 8, 2.404 s/step (~3.3 h wall, ~27 GPU-h, GPU type unstated); no success rate.
- LeRobot hardware guide (~50 demos, 5 epochs): ACT 4090 30–60 min; SmolVLA A100 1–2 h; π0/π0.5 A100 4–8 h (±50%).
- openpi DROID full-FT config comment: "100k steps … ~2 days on 8x H100" (~384 H100-h).

## Synthesis

**Lift per GPU-hour (rows reporting both):**
| Class | Representative | Order of magnitude |
|---|---|---|
| Small model, single-task imitation (scratch) | ACT +74 / ~5 GPU-h (2080 Ti); Octo +52 / ~5 A5000-h | ~10–15 pp / GPU-h |
| Real-world RL (small nets) | HIL-SERL +50 / 1–2.5 h 4090; ConRFT +57 / 45–90 min | GPU not the bottleneck; robot time + human intervention are |
| 7B VLA single-task SFT | OpenVLA full +24 / 40–120 A100-h; LoRA similar at 10–15 A100-h | ~0.2–2 pp / A100-h |
| VLA sim RL (with hours) | RL4VLA +16 to +46 / 42 A100-h; VLA-RL +4.5 / 48 GPU-h; World-Env +4.75 / ~384 H20-h | ~0.01–1 pp / GPU-h |
| Foundation-scale post-training | Cosmos Policy ~384 H100-h; CogACT 120 A100-h; MolmoAct2 YAM ~2,300 H100-h; RDT ~3,456 H100-h | hundreds–thousands H100-h |
| Synthetic data | DreamGen RoboCasa +5.9, ~81,000 L40-h for generation alone | most expensive |

Our base MolmoAct2-BimanualYAM is itself a ~2,300 H100-h YAM fine-tune (A18); our incremental L40S hours can be placed next to that number.

**Comparability to "same base 0% → X% on new hard bimanual tasks":**
- Most comparable (sim, bimanual, low start): SimpleVLA-RL / RLinf on RoboTwin 2.0 (per-task 3–17% → 64–96%, but "before" is a 1,000-demo SFT checkpoint, in-distribution seeds, no hours); RoboTwin 2.0 benchmark (π0 single-task SFT 46.4/16.3); DexMimicGen (0.7→97, small DP + 1,000 generated demos); Moonlake (YAM + π0.5 real progress +39; progress score, self-reported, no compute).
- Medium (real, new tasks, but "before" = scratch or another method): ACT, Octo, RDT, TinyVLA, Mobile ALOHA, Cosmos Policy, OFT+, MolmoAct2 YAM, Gemini specialisation, π0 complex tasks, ResFiT, GR-RL, HIL-SERL. Only π0's complex tasks are truly "same base zero-shot = 0" (TP 0→0.64/0.70). LingBot GM-100: hard new bimanual real tasks, SOTA 13–17% after 130-demo post-training.
- Not comparable: LIBERO (saturated), SimplerEnv in-distribution, open-loop MSE only (all Robocurve), relative-only numbers.
- Caveats: progress/staged scores ≠ binary success (π0, OFT+, Cosmos Policy, Moonlake, MolmoAct); 10–50 trials typical, 95% CI ~±14 (n=50) to ±30 (n=10) pp; figure-read numbers ±2 pp.

**GPU conversion:** BF16 dense TFLOPS L40S 362, A100 312, H100 SXM 989; memory bandwidth L40S 864 GB/s, A100-80 2,039 GB/s, H100 3.35 TB/s. Measured H100 ≈ 3.15× A100 (Stas Bekman MAMF). NVIDIA "L40S trains 1.7× A100" (GPT-40B LoRA, precision unstated). Practical: **1 H100-h ≈ 2.5–4 L40S-h, median ~3.** L40S has no NVLink; multi-node FSDP suffers. We report raw L40S hours and an H100-equivalent range.

## Not verified / discrepancies
1. dreamzero-yam: model card says 8×B200 ~78 h, direct $5.1–5.5K, all-in ~$6.5K (README mentions a $5K cap; H100 only as an example config).
2. "OpenVLA LIBERO 8×A100 10–15 h" not found; 8×A100 × 5–15 h is Franka full FT.
3. SmolVLA "1 A100 ~4 h / 20k steps" is LeRobot docs only.
4. Fine-tune hours not public: π0/π0.5/π*0.6 (except openpi DROID config comment), GR00T N1.5/N1.6/N1.7 (except Isaac Lab-Arena), OpenVLA-OFT, SimpleVLA-RL, RLinf, πRL, WMPO.
5. MolmoAct2 YAM per-task numbers not extracted; "15% above runner-up" abs/rel unclear; fine-tune data size unstated.
6. Moonlake is a company blog (no absolute baseline progress, no compute); π*0.6 numbers read off figures.
7. StationeryBench page (GPT-6 Astra 7/100 vs MolmoAct2 0/100) not re-verified by this survey — verify on robocurve.github.io/stationerybench before citing.
9. GPU conversion sources of mixed quality (NVIDIA 1.7× precision unstated; Lambda 2.5–3.1× seen only in a search snippet; CUDO "L40S ≈ 0.6× A100" low confidence).
10. Internal inconsistencies: SimpleVLA-RL text vs table (table used); RLinf vs SimpleVLA-RL baselines for the same checkpoint (3.13 vs 10.1); GR00T N1 DexMG numbers differ between two tables; LingBot 261 samples/s per-GPU vs 8-GPU unclear.
11. Cosmos Policy compute from the HF card, not the paper; Octo-small parameter counts from prior knowledge.
12. Venue names mostly unchecked (arXiv ids checked).

## Addendum (final report, 2026-11-09)
Additional rows:
- GR-3 (2507.15493): bimanual mobile platform real; 10 human VR trajectories per object, extra 20k steps co-training; same-base zero-shot 57.8 → 86.7 (unseen objects); compute not reported. P
- ProphRL (2511.20633): VLA-Adapter / π0.5 on SimplerEnv-WidowX; 23.3→41.0, π0.5 38.9→51.0; world-model pretrain 64×H200, RL 8×H200, hours not reported. P
- VLAC (2509.15937): 2B actor + 8B critic, real single-arm, ~200 episodes; ~30 → ~90; not reported. P (abstract-level)
- π*0.6 / Recap also reads: diverse laundry ~45→74 (figure).
- Robocurve gr00t-n1.7-yam-molmoact2 (HF card): action head + projector, 10k steps, MSE 0.00279; B200-class, count/hours/cost not recorded.
- GPU spec ratios: L40S/A100 compute 1.16×, bandwidth 0.42×, no NVLink; H100/L40S compute 2.73×, bandwidth 3.9×. MolmoAct2 YAM fine-tune 2,300 H100-h ≈ 5,750–9,200 L40S-h (derived).
- Moonlake r=0.93: one check says 15 policies, another 12 — cite as "r=0.93 (company blog)".
- SimpleVLA-RL vs RLinf report different starting values for the same SFT checkpoint (Lift Pot 10.1 vs 3.13); do not compare their absolute values.

Notes:
- Report raw L40S hours plus an H100-equivalent range (2.5–4×); B200 hours are not converted.
- A from-scratch ACT / DP baseline on the same replica would separate the contribution of the pretrained base (not included in this release).
