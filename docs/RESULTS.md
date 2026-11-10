# Results (simulation)

All numbers: our physics-sim replica, not Robocurve's robot. Full-length episodes (eraser 3600 steps = 120 s on the
spring-return lid, ruler 2000 steps = 66.7 s), 30 Hz, `nominal2` look unless stated, unseeded policy sampling.
Intervals are Wilson 95%. Paired comparisons use the same env seeds (same layout and look) and an exact McNemar test.

## Sealed set (seeds 3000000-3000049, never used for development or tuning)

| model | task | successes | rate [95% CI] | max stage 0/1/2/3/4 | mean max stage |
|---|---|---|---|---|---|
| MolmoAct2-BimanualYAM (base) | eraser_from_box | 0/50 | 0% [0–7%] | 50 / 0 / 0 / 0 / 0 | 0.00 |
| MolmoAct2-BimanualYAM (base) | ruler_handover | 0/50 | 0% [0–7%] | 36 / 13 / 1 / 0 / 0 | 0.30 |
| fine-tuned (final round, ft12) | eraser_from_box | 18/50 | 36% [24–50%] | 0 / 9 / 23 / 0 / 18 | 2.54 |
| fine-tuned (final round, ft12) | ruler_handover | 38/50 | 76% [63–86%] | 0 / 3 / 3 / 6 / 38 | 3.58 |

Paired fine-tuned vs base (same 50 seeds): eraser +18 / −0, ruler +38 / −0; exact McNemar p = 7.6e-6 and 7.3e-12.

## Development set (seeds 1000-1019), final round

| model | look | eraser_from_box (spring lid) | ruler_handover |
|---|---|---|---|
| fine-tuned (ft12) | nominal | 8/20 | 14/20 |
| fine-tuned (ft12) | nominal2 | 4/20 | 14/20 |

The development seeds were used to steer the project (data and recipe decisions), so they are optimistic as an estimate;
the sealed set above is the headline. Earlier rounds evaluated the eraser on a rigid lid before we switched to the
spring-return lid (a harder and more faithful model of the real box); numbers on the two lid models are not comparable.

## Reference: the real robot

Robocurve's published real-robot StationeryBench comparison ([2026-09-10](https://openai.robocurve.org/stationerybench/))
ran MolmoAct2 zero-shot ("no fine-tuning on these tasks or objects"): 0/20 on eraser_from_box (mean progress 0) and
0/20 on ruler_handover (mean progress 16), 0/100 over five tasks; GPT-6 Astra scored 0/20 and 1/20 on these two tasks
(7/100 overall). They report that in 47 of 100 MolmoAct2 trials the arms never left the start pose, including every
eraser trial. Our simulated base result (0/50 on both tasks, never past stage 0 on the eraser) reproduces that failure.
Sim and real numbers are not interchangeable; our fine-tuned results have not been tested on a real robot.
