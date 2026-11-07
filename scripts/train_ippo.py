"""Independent PPO with parameter sharing (IPPO / CTDE-free) for RWARE.

All robots share one actor-critic network (each robot feeds its own local observation).
Usage: python train_ippo.py --reward {sparse,shaped} --seed 0 --steps 2000000
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from common import make_train_env, stack, evaluate


p = argparse.ArgumentParser()
p.add_argument("--reward", default="shaped", choices=["sparse", "shaped"])
p.add_argument("--aug", type=int, default=1)   # task-assignment observation features
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=2_000_000)   # environment (joint) steps
p.add_argument("--n_envs", type=int, default=8)
p.add_argument("--rollout", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--mb", type=int, default=512)
p.add_argument("--eval_every", type=int, default=50_000)
p.add_argument("--eval_eps", type=int, default=10)
p.add_argument("--out", default="results")
args = p.parse_args()


torch.set_num_threads(1)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
tag = f"ippo_{args.reward}_aug{args.aug}_s{args.seed}"


class AC(nn.Module):
    def __init__(s, obs_dim, n_act, h=128):
        super().__init__()
        s.body = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        s.pi = nn.Linear(h, n_act)
        s.v = nn.Linear(h, 1)

    def forward(s, x):
        z = s.body(x)
        return s.pi(z), s.v(z).squeeze(-1)


envs = [make_train_env(args.reward, args.aug) for _ in range(args.n_envs)]
obs = np.stack([stack(e.reset(seed=args.seed * 1000 + i)[0]) for i, e in enumerate(envs)])  # (E, A, D)
E, A, D = obs.shape
n_act = envs[0].action_space[0].n
net = AC(D, n_act)
opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)


def policy_fn(greedy=False):
    def f(o):
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(o))
            d = Categorical(logits=logits)
            return (logits.argmax(-1) if greedy else d.sample()).numpy()
    return f


T = args.rollout
n_updates = args.steps // (T * E)
log = []
ep_ret, ep_ret_hist = np.zeros(E), []
t0, steps = time.time(), 0

for upd in range(n_updates):
    for g in opt.param_groups:                       # linear LR decay
        g["lr"] = args.lr * (1 - upd / n_updates)
    B_obs = np.zeros((T, E, A, D), np.float32); B_act = np.zeros((T, E, A), np.int64)
    B_lp = np.zeros((T, E, A), np.float32); B_val = np.zeros((T, E, A), np.float32)
    B_rew = np.zeros((T, E, A), np.float32); B_done = np.zeros((T, E), np.float32)
    B_boot = np.zeros((T, E, A), np.float32)  # gamma*V(s_final) added to the reward at time-limit truncation
    for t in range(T):
        with torch.no_grad():
            logits, val = net(torch.as_tensor(obs.reshape(E * A, D)))
            d = Categorical(logits=logits)
            a = d.sample()
            lp = d.log_prob(a)
        a_np = a.numpy().reshape(E, A)
        B_obs[t], B_act[t] = obs, a_np
        B_lp[t], B_val[t] = lp.numpy().reshape(E, A), val.numpy().reshape(E, A)
        for i, e in enumerate(envs):
            o, r, term, trunc, _ = e.step(list(a_np[i]))
            B_rew[t, i] = r
            ep_ret[i] += np.sum(r)
            done = term or trunc
            B_done[t, i] = done
            if done:                                   # 500-step time limit = truncation, not a real terminal
                with torch.no_grad():
                    B_boot[t, i] = args.gamma * net(torch.as_tensor(stack(o)))[1].numpy()
                ep_ret_hist.append(ep_ret[i]); ep_ret[i] = 0
                o, _ = e.reset()
            obs[i] = stack(o)
        steps += E
    with torch.no_grad():
        _, last_val = net(torch.as_tensor(obs.reshape(E * A, D)))
    last_val = last_val.numpy().reshape(E, A)

    # GAE (per agent, episode boundaries shared across the team)
    adv = np.zeros_like(B_rew); lastgae = 0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else B_val[t + 1]
        nd = B_done[t][:, None]
        delta = B_rew[t] + B_boot[t] + args.gamma * nv * (1 - nd) - B_val[t]
        lastgae = delta + args.gamma * args.lam * (1 - nd) * lastgae
        adv[t] = lastgae
    ret = adv + B_val

    f_obs = torch.as_tensor(B_obs.reshape(-1, D)); f_act = torch.as_tensor(B_act.reshape(-1))
    f_lp = torch.as_tensor(B_lp.reshape(-1)); f_adv = torch.as_tensor(adv.reshape(-1))
    f_ret = torch.as_tensor(ret.reshape(-1))
    N = f_obs.shape[0]
    for _ in range(args.epochs):
        perm = torch.randperm(N)
        for s in range(0, N, args.mb):
            idx = perm[s:s + args.mb]
            logits, v = net(f_obs[idx]); d = Categorical(logits=logits)
            ratio = (d.log_prob(f_act[idx]) - f_lp[idx]).exp()
            ad = f_adv[idx]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            pg = -torch.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
            vl = 0.5 * (v - f_ret[idx]).pow(2).mean()
            loss = pg + 0.5 * vl - args.ent * d.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

    if steps % args.eval_every < T * E or upd == n_updates - 1:
        m, sd, _ = evaluate(policy_fn(), args.eval_eps, seed0=50_000, augment=bool(args.aug))
        tr = float(np.mean(ep_ret_hist[-E:])) if ep_ret_hist else 0.0
        log.append((steps, m, sd, tr))
        print(f"{tag} steps={steps} eval_deliveries={m:.2f}±{sd:.2f} train_ret={tr:.2f} "
              f"{steps / (time.time() - t0):.0f} sps", flush=True)
        with open(f"{args.out}/{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["steps", "eval_mean", "eval_std", "train_return"])
            w.writerows(log)

torch.save(net.state_dict(), f"{args.out}/{tag}.pt")
m, sd, scores = evaluate(policy_fn(), 50, seed0=90_000, augment=bool(args.aug))
mg, sdg, scores_g = evaluate(policy_fn(greedy=True), 50, seed0=90_000, augment=bool(args.aug))
np.savez(f"{args.out}/{tag}_final.npz", stoch=scores, greedy=scores_g)
print(f"FINAL {tag}: stochastic {m:.2f}±{sd:.2f} | greedy {mg:.2f}±{sdg:.2f}", flush=True)