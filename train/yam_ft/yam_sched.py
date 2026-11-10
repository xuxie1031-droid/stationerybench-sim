"""Extra LR schedules for train_yam_ft.py (opt-in via env; default = the stock per-group cosine-with-warmup).

The trainer applies lr(step) with step = global_step, which is 1 for the first optimizer update and max_steps for
the last one.  Group base lrs are the usual --llm/--vit/--connector/--action_expert_learning_rate values.

YAM_FT_SCHED=cont   continuation (phase 2): no warmup; every group starts at its own base lr at step 1 and decays
                    linearly to CONT_END_FRAC (default 0.01) x base at the last step:
                    lr(s) = base * (1 - (1 - end) * (s - 1) / (S - 1))
YAM_FT_SCHED=wsd    warmup-stable-decay: per-group linear warmup over its --scheduler.<group>_t_warmup steps from
                    warmup_min_lr (trainer default 0) to base, constant base, then decay over the last
                    WSD_DECAY_FRAC (default 0.2) of the steps to WSD_END_FRAC (default 0) x base; WSD_SHAPE=linear
                    (default) or sqrt (1-sqrt decay).  Decay start D = S - round(frac * S); lr(D) = base.
"""
import math, os
from dataclasses import dataclass, field
from typing import Dict, Optional

import olmo.train.optim as O

GROUP_FIELDS = {"connector": "connector_t_warmup", "vit": "vit_t_warmup", "llm": "llm_t_warmup",
                "frame_selector": "frame_selector_t_warmup", "temporal_token_scorer": "temporal_token_scorer_t_warmup",
                "action_expert": "action_expert_t_warmup"}


@dataclass
class YamScheduler(O.Scheduler):
    mode: str = "cont"
    warmups: Dict[str, int] = field(default_factory=dict)
    cont_end: float = 0.01
    decay_frac: float = 0.2
    end_frac: float = 0.0
    shape: str = "linear"

    def _warm(self, group_name):
        for k, w in self.warmups.items():
            if group_name and group_name.startswith(k):
                return w
        return 0

    def get_lr(self, initial_lr: float, step: int, max_steps: int, group_name: str = None) -> float:
        S = max(int(max_steps), 1); s = min(max(int(step), 0), S)
        if self.mode == "cont":
            if S <= 1:
                return initial_lr
            return initial_lr * (1.0 - (1.0 - self.cont_end) * max(s - 1, 0) / (S - 1))
        # wsd
        W = self._warm(group_name)
        D = S - int(round(self.decay_frac * S))
        if W > 0 and s < W:
            lo = self.warmup_min_lr if self.warmup_min_lr is not None else 0.0
            return lo + (initial_lr - lo) * s / W
        if s <= D or D >= S:
            return initial_lr
        x = (s - D) / (S - D)
        f = x if self.shape == "linear" else math.sqrt(x)
        return initial_lr * (self.end_frac + (1.0 - self.end_frac) * (1.0 - f))


def make(cfg) -> YamScheduler:
    mode = os.environ["YAM_FT_SCHED"]
    if mode not in ("cont", "wsd"):
        raise ValueError(f"YAM_FT_SCHED={mode!r}: expected cont or wsd")
    return YamScheduler(
        grad_clip_warmup_steps=None if cfg.grad_clip_warmup_steps is None else int(cfg.grad_clip_warmup_steps),
        grad_clip_warmup_factor=cfg.grad_clip_warmup_factor, warmup_min_lr=cfg.warmup_min_lr, mode=mode,
        warmups={k: int(getattr(cfg, f)) for k, f in GROUP_FIELDS.items()},
        cont_end=float(os.environ.get("CONT_END_FRAC") or 0.01),
        decay_frac=float(os.environ.get("WSD_DECAY_FRAC") or 0.2),
        end_frac=float(os.environ.get("WSD_END_FRAC") or 0.0),
        shape=os.environ.get("WSD_SHAPE") or "linear")


def install():
    O.SchedulerConfig.build = lambda self: make(self)
    print(f"yam_ft scheduler: {os.environ['YAM_FT_SCHED']} (CONT_END_FRAC={os.environ.get('CONT_END_FRAC')}, "
          f"WSD_DECAY_FRAC={os.environ.get('WSD_DECAY_FRAC')}, WSD_END_FRAC={os.environ.get('WSD_END_FRAC')}, "
          f"WSD_SHAPE={os.environ.get('WSD_SHAPE')})", flush=True)
