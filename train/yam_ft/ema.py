"""EMA of the trainable weights for train_yam_ft.py (opt-in: YAM_FT_EMA=<decay>, e.g. 0.999).

- Shadow = fp32 copy of the LOCAL tensor of every requires_grad parameter (DTensor.to_local(); with replicate-only
  HSDP / single GPU that is the full tensor, with full-shard FSDP it is this rank's shard).  Device: YAM_FT_EMA_DEVICE
  (default cuda; 'cpu' if memory is short).  Frozen parameters are not copied.
- Update after every optimizer step:  shadow = decay * shadow + (1 - decay) * param   (torch._foreach_mul_/add_).
  decay=0 reproduces the live weights and decay=1 the initial weights bitwise (x*0+p == p, s*1+p*0 == s).
  Works unchanged with YAM_FT_ACCUM_NOSYNC (params are identical on all replicas after the synced step).
- Saving: right after every regular checkpoint the shadow is swapped into the model and saved to
  <save_folder>/ema/step<N>/ (AE/full FT: a normal sharded model checkpoint, no optimizer, plus shadow_rank<r>.pt for
  resume) and, for LoRA runs, <save_folder>/ema/step<N>-merged (+ -lora-llm/-vision) via the trainer's own LoRA
  save; then the live weights are swapped back.  Only the latest EMA step is kept.
  Export/serve: EMA=1 yam_ft/serve_ckpt.sh runs/<NAME>/step<N>[-merged]  -> uses ema/step<N>[-merged].
- Resume: if <save_folder>/ema/step<global_step>/shadow_rank<r>.pt exists the shadow is restored, else re-initialised
  from the current weights (warning).  YAM_FT_EMA_INIT=<.../shadow_rank{rank}.pt> continues another run's EMA at
  step 0 (names/shapes must match, e.g. AE phase 1 -> phase 2; not possible across LoRA runs, see ft2_plan.sh).
  YAM_FT_EMA_INIT may also be an EMA checkpoint directory (<run>/ema/step<N>): the trainable tensors are then read
  from its model_and_optim (needed for runs saved before 11-04 19:00, whose AE shadow files were overwritten).
- Test hooks: YAM_FT_EMA_TRACE=k records k elements of 3 tensors after every update for the closed-form check
  (printed at the end of fit as EMA_CLOSED_FORM max_rel_err=...); EMA_HASH rank=r sha256=... is printed at the end.
"""
import glob, hashlib, logging, os, shutil
import torch

log = logging.getLogger(__name__)


class EMA:
    def __init__(self, named_params, decay, device="cuda", trace=0):
        self.decay = float(decay)
        self.names, self.params = [], []
        for n, p in named_params:
            if p.requires_grad:
                self.names.append(n); self.params.append(p)
        self.device = device
        self.shadow = [self._local(p).detach().to(device=device, dtype=torch.float32, copy=True) for p in self.params]
        self.num_updates = 0
        self.trace = int(trace)
        if self.trace:
            self.trace_ix = [0, len(self.params) // 2, len(self.params) - 1]
            self.trace_init = [self.shadow[i].reshape(-1)[: self.trace].double().cpu().clone() for i in self.trace_ix]
            self.trace_live = []

    @staticmethod
    def _local(p):
        return p.to_local() if hasattr(p, "to_local") else p

    @torch.no_grad()
    def update(self):
        live = [self._local(p).detach() for p in self.params]
        if self.device != "cuda" and live and live[0].is_cuda:
            live = [x.to(self.device, non_blocking=False) for x in live]
        live = [x if x.dtype == torch.float32 else x.float() for x in live]
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, live, alpha=1.0 - self.decay)
        self.num_updates += 1
        if self.trace:
            self.trace_live.append([live[i].reshape(-1)[: self.trace].double().cpu().clone() for i in self.trace_ix])

    @torch.no_grad()
    def swap(self):
        """Exchange shadow and live weights (call twice to restore)."""
        for p, s in zip(self.params, self.shadow):
            loc = self._local(p)
            tmp = loc.detach().to(device=s.device, dtype=torch.float32, copy=True)
            loc.copy_(s.to(device=loc.device, dtype=loc.dtype))
            s.copy_(tmp)

    def sha256(self):
        # same encoding/order as train_yam_ft.py's PARAM_HASH (sorted names, fp32 bytes) -> directly comparable
        h = hashlib.sha256()
        for n, s in sorted(zip(self.names, self.shadow), key=lambda x: x[0]):
            h.update(n.encode()); h.update(s.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates, "names": self.names,
                "shadow": [s.detach().cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        assert sd["names"] == self.names, "EMA param names differ"
        for s, t in zip(self.shadow, sd["shadow"]):
            s.copy_(t.to(s.device))
        self.num_updates = int(sd["num_updates"])

    def closed_form_check(self):
        """e_T = d^T e_0 + (1-d) sum_k d^(T-k) p_k  (float64)  vs the fp32 shadow."""
        d, T, worst = self.decay, len(self.trace_live), 0.0
        for j, i in enumerate(self.trace_ix):
            e = (d ** T) * self.trace_init[j]
            for k in range(T):
                e = e + (1 - d) * (d ** (T - 1 - k)) * self.trace_live[k][j]
            got = self.shadow[i].reshape(-1)[: self.trace].double().cpu()
            worst = max(worst, float(((got - e).abs() / e.abs().clamp_min(1e-6)).max()))
        return T, worst


def shadow_from_dcp(ckpt_dir, ema):
    """Build an EMA state dict from a saved (EMA) model checkpoint: <ckpt_dir>/model_and_optim (torch DCP, keys
    'model.<param name>').  Works when this rank's local tensors are full tensors (single GPU / replicate-only HSDP);
    returns None if keys or shapes do not match.  Collective: every rank must call it."""
    import torch.distributed.checkpoint as dcp
    from olmo.train.remote_filesystem import RemoteFileSystemReader  # olmo writes its own storage format
    path = os.path.join(ckpt_dir, "model_and_optim")
    md = RemoteFileSystemReader(path).read_metadata().state_dict_metadata
    want = {}
    for n, s in zip(ema.names, ema.shadow):
        # runtime names carry wrapper prefixes (activation checkpointing / FSDP / compile) that state dicts drop
        k = "model." + n.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        if k not in md or tuple(md[k].size) != tuple(s.shape):
            print(f"shadow_from_dcp: {k} missing or shape {getattr(md.get(k), 'size', None)} != {tuple(s.shape)}", flush=True)
            return None
        want[k] = torch.empty(tuple(s.shape), dtype=torch.float32)
    dcp.load(want, storage_reader=RemoteFileSystemReader(path))
    keys = ["model." + n.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
            for n in ema.names]
    return {"decay": ema.decay, "num_updates": 0, "names": ema.names, "shadow": [want[k] for k in keys]}


def install():
    import torch.distributed as dist
    import olmo.train.trainer as T
    from olmo.train.trainer import CheckpointType
    decay = float(os.environ["YAM_FT_EMA"])
    device = os.environ.get("YAM_FT_EMA_DEVICE") or "cuda"   # launchers pass empty strings for unset vars
    trace = int(os.environ.get("YAM_FT_EMA_TRACE") or 0)
    orig_fit, orig_save = T.Trainer.fit, T.Trainer.save_checkpoint
    rank = lambda: dist.get_rank() if dist.is_initialized() else 0
    barrier = lambda: dist.barrier() if dist.is_initialized() else None

    def ema_dir(self, step):
        return os.path.join(str(self.cfg.save_folder), "ema", f"step{step}")

    def fit(self):
        self._yam_ema = EMA(self.fsdp_model.named_parameters(), decay, device=device, trace=trace)
        sp = os.path.join(ema_dir(self, self.global_step), f"shadow_rank{rank()}.pt")
        init_p = (os.environ.get("YAM_FT_EMA_INIT") or "").replace("{rank}", str(rank()))
        if self.global_step == 0 and init_p:
            # continue the EMA of a previous run (e.g. phase 1 -> phase 2 with the same trainable parameter set)
            if os.path.isdir(init_p):
                sd = shadow_from_dcp(init_p, self._yam_ema)
            else:
                sd = torch.load(init_p) if os.path.exists(init_p) else None
            if sd is None or sd["names"] != self._yam_ema.names or \
                    any(a.shape != b.shape for a, b in zip(sd["shadow"], self._yam_ema.shadow)):
                # never kill a long run over this: fall back to a fresh EMA and say so loudly
                print(f"EMA_INIT_MISMATCH {init_p}: missing or different trainable parameter set -> fresh EMA", flush=True)
            else:
                self._yam_ema.load_state_dict(sd)
                print(f"EMA continued from {init_p} (updates so far {sd['num_updates']})", flush=True)
        elif self.global_step > 0:
            if os.path.exists(sp):
                self._yam_ema.load_state_dict(torch.load(sp)); log.info("EMA restored from %s", sp)
            else:
                log.warning("EMA: resumed at step %d without %s -> shadow re-initialised", self.global_step, sp)
        print(f"EMA_INIT_HASH rank={rank()} sha256={self._yam_ema.sha256()}", flush=True)
        live = EMA(self.fsdp_model.named_parameters(), decay, device="cpu")  # same encoding, current live weights
        lh = live.sha256(); del live
        print(f"LIVE_INIT_HASH rank={rank()} sha256={lh} | EMA shadow == live weights at start: {lh == self._yam_ema.sha256()}", flush=True)
        n = sum(s.numel() for s in self._yam_ema.shadow)
        log.info("EMA on: decay %s, %d trainable tensors, %d local elements, device %s", decay, len(self._yam_ema.shadow), n, device)
        opt = self.optim
        if not getattr(opt, "_yam_ema_wrapped", False):
            orig_step = opt.step

            def step(*a, **k):
                out = orig_step(*a, **k)
                self._yam_ema.update()
                return out
            opt.step = step; opt._yam_ema_wrapped = True
        out = orig_fit(self)
        print(f"EMA_HASH rank={rank()} updates={self._yam_ema.num_updates} sha256={self._yam_ema.sha256()}", flush=True)
        if trace:
            Tn, err = self._yam_ema.closed_form_check()
            print(f"EMA_CLOSED_FORM rank={rank()} steps={Tn} decay={decay} max_rel_err={err:.3e}", flush=True)
        return out

    def save_checkpoint(self, checkpoint_type, optim=True):
        out = orig_save(self, checkpoint_type, optim)
        if checkpoint_type != CheckpointType.sharded or not hasattr(self, "_yam_ema"):
            return out
        step = self.global_step; d = ema_dir(self, step)
        os.makedirs(d, exist_ok=True)
        self._yam_ema.swap()
        try:
            if self.cfg.model.lora_enable:
                sf = self.cfg.save_folder
                self.cfg.save_folder = os.path.join(str(sf), "ema")
                try:
                    self.save_lora_checkpoint(save_and_remove=False)
                finally:
                    self.cfg.save_folder = sf
            else:
                self.optim.zero_grad(set_to_none=True)
                barrier()
                self.checkpointer.save(d, self.fsdp_model, None, self.trainer_state_dict(), config=self.cfg)
        finally:
            self._yam_ema.swap()
        barrier()
        # after checkpointer.save: it replaces the step directory, so an earlier shadow file would be deleted
        torch.save(self._yam_ema.state_dict(), os.path.join(d, f"shadow_rank{rank()}.pt"))
        barrier()
        if rank() == 0:  # keep only the latest EMA step
            for old in glob.glob(os.path.join(str(self.cfg.save_folder), "ema", "step*")):
                if not os.path.basename(old).startswith(f"step{step}"):
                    shutil.rmtree(old, ignore_errors=True)
        log.info("EMA checkpoint saved to %s", d)
        return out

    T.Trainer.fit, T.Trainer.save_checkpoint = fit, save_checkpoint
