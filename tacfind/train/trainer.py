"""Trainer: one loop for every training scheme. Per iteration: collect a rollout with the stage's actor (scripted
explorer or actor-critic), then optimise the weighted sum of the stage's enabled losses over minibatches.

Everything lives on the env device; per iteration only the logged scalars, and the one rendered world, reach the CPU.
"""

import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import DictConfig

from tacfind.envs.bin_env import BinEnv
from tacfind.envs.explorer import Explorer
from tacfind.networks.model import Percept, TacFind
from tacfind.train.logger import RunLogger, gpu_mem_gb, rates, to_floats
from tacfind.train.losses import chamfer_loss, confidence_loss, contrastive_loss, ppo_loss, uniformity_loss
from tacfind.train.rollout import Rollout, gae
from tacfind.train.stages import Stage, matcher_inputs, overlaps, parse_stages
from tacfind.train.viz import Clip, plot_curves

PERCEPTION = ("contrastive", "confidence", "chamfer")


def resolve_device(device: str) -> str:
    """`auto` -> cuda when available, else cpu."""
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def load_checkpoint(path: str | Path, device) -> dict:
    """A checkpoint file, or a run directory (-> its ckpt/last.pt)."""
    path = Path(path)
    return torch.load(path / "ckpt" / "last.pt" if path.is_dir() else path, map_location=device)


def random_rotation(n: int, device) -> torch.Tensor:
    """(n, 3, 3) uniformly random rotation matrices (QR of a Gaussian matrix, signs fixed, det +1)."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, device=device))
    q = q * torch.diagonal(r, dim1=-2, dim2=-1).sign()[:, None]
    return q * torch.stack([torch.det(q), torch.ones(n, device=device), torch.ones(n, device=device)], -1)[:, None]


class Trainer:
    """Environment, explorer, model, rollout storage and the stage loop of one run.

    Args:
        cfg: resolved root config. logger: the run directory. split: library split of the worlds ("train" |
            "heldout").
    """

    def __init__(self, cfg: DictConfig, logger: RunLogger, split: str = "train"):
        self.cfg, self.log = cfg, logger
        torch.manual_seed(cfg.seed)
        self.env = env = BinEnv(cfg.env, cfg.n_envs, cfg.history, resolve_device(cfg.device), cfg.seed, split)
        self.dev, self.N, self.T, self.H = env.device, cfg.n_envs, cfg.rollout_steps, cfg.history
        self.explorer = Explorer(env, cfg.seed)
        self.model = TacFind(cfg, env.spec).to(self.dev)
        if cfg.init_from:
            self.model.load_state_dict(load_checkpoint(cfg.init_from, self.dev)["model"])
        # contrastive library = this split's objects only: held-out shapes are never seen in training
        self.clouds = env.clouds[env.lib_ids]
        self.lib_pos = torch.full((len(env.clouds),), -1, dtype=torch.long, device=self.dev)
        self.lib_pos[env.lib_ids] = torch.arange(len(env.lib_ids), device=self.dev)
        self.buf = Rollout(self.N, self.T, self.H, len(env.clouds), self.dev)
        self.obs = env.reset()
        self._store_obs(self.T + self.H - 1, self.obs, None)  # the "next" slot: `advance` moves it to step 0
        self.ep_ret, self.ep_len = torch.zeros(self.N, device=self.dev), torch.zeros(self.N, device=self.dev)
        self.env_steps, self.clip, self.deterministic, self.recompute = 0, None, False, (True, True)
        self.augment = cfg.loss.augment_rotate  # off in evaluation
        self.live = None
        if cfg.live_port is not None:  # spectate world viz_world in the browser (see tacfind/viz/live.py)
            from tacfind.viz.live import LiveViewer
            self.live = LiveViewer(env, ("training",), cfg.live_port, title=f"tacfind {cfg.name}").start()
            self.live._perturb = lambda world: None  # browser pushes never reach the training worlds
            self.live.bridge.cmd.world = cfg.viz_world
            print(f"live view: http://localhost:{self.live.port}/ (spectating only; tunnel the port)", flush=True)
        gpu = torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else None
        logger.write_summary(device=str(self.dev), gpu_name=gpu, n_envs=self.N, params=self.model.param_counts())

    # ------------------------------------------------------------------------------------------------ stages --
    def run(self) -> None:
        """Run every selected stage of the training scheme."""
        cfg = self.cfg
        for stage in parse_stages(cfg.train, self.model, cfg.optim, cfg.run_stages, cfg.max_iters):
            self.run_stage(stage)

    def run_stage(self, stage: Stage) -> None:
        """Collect / update for `stage.iters` iterations; checkpoint, video and curves at the end."""
        cfg, model = self.cfg, self.model
        params = model.set_trainable(stage.train)
        opt = torch.optim.Adam(params, lr=stage.lr)
        self.recompute = (overlaps(stage.train, ["touch_encoder"]), overlaps(stage.train, matcher_inputs(model)))
        clip_start = max(0, stage.iters - math.ceil(cfg.viz_steps / self.T))  # the last clip ends with the stage
        for it in range(stage.iters):
            if self.clip is None and (it % cfg.viz_every == 0 or it == clip_start):
                self.clip = Clip(self.env, cfg.viz_world, self.log.dir / "videos" / f"{stage.name}_{it:05d}.mp4",
                                 cfg.viz_steps)
            t0 = time.time()
            roll = self.collect(stage, it)
            t1 = time.time()
            upd = self.update(stage, opt, params)
            self.env_steps += self.N * self.T
            if it % cfg.log_every == 0 or it == stage.iters - 1:
                self._log(stage, it, rates(roll) | upd, t1 - t0, time.time() - t0)
            if (it + 1) % cfg.ckpt_every == 0:
                self.save("last", stage, it)
            if it % cfg.viz_every == 0 and it < stage.iters - 1:  # the last iteration plots below
                plot_curves(self.log.records, self.log.dir / "figures" / "curves.png")
        self._finish_clip()
        self.save(stage.name, stage, stage.iters - 1)
        self.save("last", stage, stage.iters - 1)
        plot_curves(self.log.records, self.log.dir / "figures" / "curves.png")

    def close(self) -> None:
        """Stop the live viewer and the cpu backend's threads."""
        if self.live:
            self.live.close()
        if hasattr(self.env.sim, "pool"):
            self.env.sim.pool.shutdown()

    def save(self, tag: str, stage: Stage, it: int) -> None:
        """ckpt/<tag>.pt: model weights and where training stood."""
        torch.save({"model": self.model.state_dict(), "stage": stage.name, "iter": it, "env_steps": self.env_steps},
                   self.log.dir / "ckpt" / f"{tag}.pt")

    def _log(self, stage: Stage, it: int, scalars: dict, t_roll: float, t_iter: float) -> None:
        rec = to_floats(scalars, self.dev) | {"perf/env_steps_per_s": self.N * self.T / t_roll, "perf/iter_s": t_iter,
                                          "perf/gpu_mem_gb": gpu_mem_gb()}
        rec = self.log.log(self.env_steps, it, stage.name, rec)
        show = ("loss/contrastive", "acc/contrastive_top1", "loss/confidence", "conf/mae", "loss/policy",
                "rollout/return", "rollout/success_rate", "perf/env_steps_per_s", "perf/gpu_mem_gb")
        print(f"[{stage.name} {it}] step {self.env_steps} " + " ".join(
            f"{k.split('/')[1]}={rec[k]:.4g}" for k in show if rec.get(k) is not None), flush=True)

    # ---------------------------------------------------------------------------------------------- rollout --
    def _store_obs(self, k: int, obs: dict, info: dict | None) -> None:
        """Observation at position k with the labels of the state it shows (none at an episode's first step)."""
        if info is None:  # after env.reset every world is at its first step: shape-only placeholders, masked below
            info = {"tip_obj": obs["pts_valid"].long(), "conf_gt": obs["first"].float(),
                    "priv": torch.zeros(self.N, self.env.spec.priv_dim, device=self.dev)}
        keep = ~obs["first"]
        self.buf.add_obs(k, o=obs["o"], pts=obs["pts"], valid=obs["pts_valid"], first=obs["first"],
                         target=obs["target"], obj=torch.where(keep[:, None], info["tip_obj"], -1).long(),
                         conf_gt=(info["conf_gt"] * keep).float(), priv=(info["priv"] * keep[:, None]).float())

    def _act(self, t: int, idx: torch.Tensor, code) -> dict:
        """Perceive + act for worlds `idx` at step t (t = T: the observation after the last step)."""
        w = self.buf.window(torch.full_like(idx, t), idx)
        p = self.model.perceive(w["obs_hist"], w["hist_mask"], w["touch_pts"], w["touch_mask"],
                                code[self.lib_pos[w["target"]]])
        dist, value = self.model.act(w["obs_hist"], p, w["priv"])
        action = dist.mean if self.deterministic else dist.sample()
        return dict(action=action, logp=dist.log_prob(action).sum(-1), value=value, z_touch=p.z_touch,
                    z_diff=p.z_diff, conf_logit=p.conf_logit)

    def _act_all(self, t: int, code) -> dict:
        """`_act` for all worlds in chunks of `perceive_chunk` (bounds the matcher's memory)."""
        outs = [self._act(t, idx, code) for idx in torch.arange(self.N, device=self.dev).split(self.cfg.perceive_chunk)]
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    @torch.no_grad()
    def collect(self, stage: Stage, it: int) -> dict:
        """T env steps with the stage's actor into `self.buf`. Returns episode / contact sums (device tensors)."""
        env, H, policy = self.env, self.H, stage.actor == "policy"
        self.model.eval()
        self.buf.advance()
        code = self.model.encode_target(self.clouds) if policy or self.clip else None
        s = {k: torch.zeros((), device=self.dev) for k in ("episodes", "ret", "success", "wrong", "len", "contact",
                                                           "touch_target")}
        out = {}
        for t in range(self.T):
            out = self._act_all(t, code) if policy else {}
            action = out["action"] if policy else self.explorer.act(self.obs)
            if self.clip:
                self._frame(t, stage, it, code, out)
            s["contact"] += self.obs["pts_valid"].any(1).float().mean()
            self.obs, reward, done, info = env.step(action)
            trunc = info["timeout"] & ~info["success"] & ~info["wrong_pick"]
            self.buf.add_step(t, reward=reward, done=done, trunc=trunc, **out)
            self._store_obs(t + H, self.obs, info)
            if self.live:  # one sync per step; driver, reset and pause commands are ignored (data stays undisturbed)
                self.live.sync()
            self.ep_ret += reward
            self.ep_len += 1
            s["episodes"] += done.sum()
            s["ret"] += (self.ep_ret * done).sum()
            s["len"] += (self.ep_len * done).sum()
            s["success"] += info["success"].sum()
            s["wrong"] += (info["wrong_pick"] & ~info["success"]).sum()
            s["touch_target"] += info["touch_target"].float().mean()
            self.ep_ret, self.ep_len = self.ep_ret * ~done, self.ep_len * ~done
        if policy:
            self.last_value = self._act_all(self.T, code)["value"]
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)  # honest rollout timing
        return s | {"steps": torch.tensor(float(self.T), device=self.dev)}

    def _frame(self, t: int, stage: Stage, it: int, code, out: dict) -> None:
        """Render world `viz_world` with ground-truth and predicted confidence; save the clip when complete."""
        i = self.cfg.viz_world
        conf_logit = out["conf_logit"][i] if out else self._act(t, torch.tensor([i], device=self.dev),
                                                               code)["conf_logit"][0]
        last = self.env.last
        touched = int(last["touched"][i]) if last and not self.obs["first"][i] else -1
        text = (f"{stage.name} it {it}  t {int(self.env.t[i])}  target {int(self.obs['target'][i])}  "
                f"touched {touched}")
        if self.clip.add(float(self.buf.pos["conf_gt"][t + self.H - 1, i]), float(torch.sigmoid(conf_logit)), text):
            self._finish_clip()

    def _finish_clip(self) -> None:
        if self.clip and (path := self.clip.save()):
            self.log.video(path, self.env_steps)
        self.clip = None

    # ----------------------------------------------------------------------------------------------- update --
    def update(self, stage: Stage, opt: torch.optim.Optimizer, params: list) -> dict:
        """Epochs of minibatch steps on the rollout; PPO stops early when the epoch's mean KL exceeds target_kl."""
        cfg, N, policy = self.cfg, self.N, stage.actor == "policy"
        adv = ret = None
        if policy:
            st = self.buf.step
            reward = st["reward"] + cfg.ppo.gamma * st["value"] * st["trunc"]  # time-out: bootstrap with V(s_t)
            adv, ret = gae(reward, st["value"], st["done"], self.last_value, cfg.ppo.gamma, cfg.ppo.lam)
        self.model.train()
        has_contact = self.buf.has_contact().flatten()  # (T * N,) windows with at least one valid touch point
        pool = has_contact.nonzero()[:, 0] if stage.contact_only else torch.arange(self.T * N, device=self.dev)
        pool = pool if len(pool) else torch.arange(self.T * N, device=self.dev)  # no contact at all: keep uniform
        terms_all, n_mb = defaultdict(list), math.ceil(len(pool) / stage.batch_size)
        for epoch in range(stage.epochs):
            mbs = pool[torch.randperm(len(pool), device=self.dev)].chunk(n_mb)
            for mb in mbs:
                t, n = mb // N, mb % N
                total, terms = self.losses(stage, t, n, adv[t, n] if policy else None, ret[t, n] if policy else None)
                opt.zero_grad(set_to_none=True)
                if total.requires_grad:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.optim.max_grad_norm)
                    opt.step()
                for k, v in terms.items():
                    terms_all[k].append(v.detach())
            if policy and torch.stack(terms_all["ppo/kl"][-len(mbs):]).mean() > cfg.ppo.target_kl:
                break  # one host sync per epoch
        terms_all["rollout/contact_window_frac"] = [has_contact.float().mean()]
        out = {k: torch.stack(v).float().nanmean() for k, v in terms_all.items()}
        return out | ({"ppo/epochs": epoch + 1} if policy else {})

    def _augment(self, clouds: torch.Tensor) -> torch.Tensor:
        """Library clouds (n, P, 6), each under a random rotation when `loss.augment_rotate`."""
        if not self.augment:
            return clouds
        rot_t = random_rotation(len(clouds), self.dev).transpose(1, 2)
        return torch.cat([clouds[..., :3] @ rot_t, clouds[..., 3:] @ rot_t], -1)

    def losses(self, stage: Stage, t: torch.Tensor, n: torch.Tensor, adv: torch.Tensor | None,
               ret: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
        """Weighted sum of the stage's enabled loss terms on samples (step t, world n).

        Matcher outputs and z_touch stored at rollout time are reused when nothing they depend on trains.
        Returns:
            total loss (scalar), logged terms (scalars).
        """
        cfg, model, L = self.cfg, self.model, stage.loss
        w = self.buf.window(t, n)
        st = {k: v[t, n] for k, v in self.buf.step.items()} if stage.actor == "policy" else {}
        perception, (new_z, new_match) = any(k in L for k in PERCEPTION), self.recompute
        new_match = perception or (new_match and "ppo" in L)
        grad_match = perception or "uniform" in L or not stage.detach_matcher  # else PPO-only: no matcher graph
        with torch.set_grad_enabled(torch.is_grad_enabled() and grad_match):
            code = model.encode_target(self._augment(self.clouds)) if new_match or "uniform" in L else None
        if new_match:
            z = model.touch_encoder(w["obs_hist"], w["hist_mask"])
            with torch.set_grad_enabled(torch.is_grad_enabled() and grad_match):
                p = Percept(z, **model.matcher(z if grad_match else z.detach(), w["touch_pts"], w["touch_mask"],
                                               code[self.lib_pos[w["target"]]]))
        elif st:
            z = model.touch_encoder(w["obs_hist"], w["hist_mask"]) if new_z else st["z_touch"]
            p = Percept(z, st["z_diff"], st["conf_logit"], None)
        terms, total = {}, torch.zeros((), device=self.dev)
        if "contrastive" in L:
            label = torch.where(w["touched"] >= 0, self.lib_pos[w["touched"].clamp(min=0)], -1)
            terms["loss/contrastive"], terms["acc/contrastive_top1"] = contrastive_loss(
                p.e_touch, code.z, label, cfg.loss.temperature)
        if "uniform" in L:
            terms["loss/uniform"] = uniformity_loss(code.z, cfg.loss.uniform_t)
        if "confidence" in L:
            terms["loss/confidence"] = confidence_loss(p.conf_logit, w["conf_gt"])
        if "chamfer" in L:  # only the latent_head matcher estimates contact points; otherwise the term is zero
            terms["loss/chamfer"] = total if p.est_points is None else chamfer_loss(
                p.est_points, w["touch_pts"][..., :3], w["touch_mask"], model.matcher.embed.tokenizer.pos_scale)
        for k in L:
            if k != "ppo":
                total = total + L[k] * terms[f"loss/{k}"]
        if "ppo" in L:
            dist, value = model.act(w["obs_hist"], p, w["priv"], stage.detach_matcher)
            pol, val, kl = ppo_loss(dist.log_prob(st["action"]).sum(-1), st["logp"],
                                    (adv - adv.mean()) / (adv.std() + 1e-8), value, ret, cfg.ppo.clip)
            ent = dist.entropy().sum(-1).mean()
            terms |= {"loss/policy": pol, "loss/value": val, "ppo/entropy": ent, "ppo/kl": kl}
            total = total + L["ppo"] * (pol + cfg.ppo.value_coef * val - cfg.ppo.entropy_coef * ent)
        if perception or st:
            conf = p.conf.detach()
            terms |= {"conf/mae": (conf - w["conf_gt"]).abs().mean(), "conf/pred_mean": conf.mean(),
                      "conf/gt_mean": w["conf_gt"].mean()}
        return total, terms
