"""Mid-run data-mix switch (opt-in):  YAM_FT_MIX_SWITCH="<step>:<real_share>"  e.g. "2800:0.6667".
From optimizer step <step> on (i.e. from example <step> x global_batch_size of the deterministic sample stream), the
per-dataset sampling rates are rescaled so that real/* datasets together get <real_share> and the sim datasets the
rest, keeping each group's internal proportions.  Before the switch the stream is bitwise the stock one.  Works with
resume (the mixture fast-forwards through the same stream).  Installed at import so dataloader workers get it."""
import os
import numpy as np


def install():
    from olmo.data.iterable_dataset_mixture import IterableDatasetMixture as M
    if getattr(M, "_yam_switch", False):
        return
    step, share = os.environ["YAM_FT_MIX_SWITCH"].split(":")
    step, share = int(step), float(share)
    orig = M._get_next_source

    def _get_next_source(self, rng, counts):
        if len(self.datasets) > 1 and self.global_batch_size and counts.sum() >= step * self.global_batch_size:
            if not hasattr(self, "_yam_rates2"):
                real = np.array([str(getattr(getattr(d, "dataset", d), "_get_repo_id", lambda: "")()).startswith("real/")
                                 for d in self.datasets])
                r = self.mixture_rates.astype(np.float64); r1 = r[real].sum()
                r2 = np.where(real, r * share / r1, r * (1 - share) / (1 - r1))
                self._yam_rates2 = (r2 / r2.sum()).astype(np.float32)
                print(f"MIX_SWITCH pid={os.getpid()} at example {int(counts.sum())} (step {step}): real share "
                      f"{r1:.4f} -> {self._yam_rates2[real].sum():.4f}", flush=True)
            return rng.choice(len(self.datasets), p=self._yam_rates2)
        return orig(self, rng, counts)

    M._get_next_source = _get_next_source
    M._yam_switch = True
