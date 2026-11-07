"""Learning curves + summary table. Metric = true deliveries/episode on the SPARSE-reward eval env."""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

rand = np.load("results/random_scores.npy"); heur = np.load("results/heuristic_scores.npy")
cfg = [  # (file glob, label, colour, linestyle)
    ("ippo_sparse_aug0", "Sparse reward, base obs", "#7f7f7f", "-"),
    ("ippo_shaped_aug0", "Shaped reward, base obs", "#c0392b", "-"),
    ("ippo_sparse_aug1", "Sparse reward + task hints", "#e69f00", "-"),
    ("ippo_shaped_aug1", "Shaped reward + task hints", "#0072b2", "-"),
]
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=170)
rows = []
for key, lab, col, ls in cfg:
    files = sorted(glob.glob(f"results/{key}_s?.csv")); runs = [pd.read_csv(f) for f in files]
    if not runs: continue
    n = min(len(r) for r in runs); x = runs[0].steps[:n] / 1e6
    Y = np.array([r.eval_mean[:n] for r in runs]); y = Y.mean(0)
    ax.plot(x, y, color=col, ls=ls, lw=2, label=f"{lab} (n={len(runs)})")
    if len(runs) > 1: ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=.18, lw=0)
    fin = [np.load(f.replace(".csv", "_final.npz"))["stoch"] for f in files if glob.glob(f.replace(".csv", "_final.npz"))]
    if fin:
        per_seed = [s.mean() for s in fin]
        rows.append((lab, len(fin), f"{np.mean(per_seed):.1f}", f"{np.concatenate(fin).std():.1f}",
                     ", ".join(f"{p:.1f}" for p in per_seed), f"{min(r.steps.iloc[-1] for r in runs)/1e6:.1f}M"))
ax.axhline(heur.mean(), color="#2e7d32", ls="--", lw=1.6, label=f"Scripted heuristic ({heur.mean():.1f})")
ax.axhline(rand.mean(), color="black", ls=":", lw=1.6, label=f"Random policy ({rand.mean():.1f})")
ax.set_xlabel("Environment steps (millions)"); ax.set_ylabel("Deliveries per 500-step episode")
ax.set_title("RWARE tiny-2ag: IPPO learning curves (band = min/max over seeds)", fontsize=10)
ax.legend(frameon=True, framealpha=.9, edgecolor="none", fontsize=7.3, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.07)); ax.grid(alpha=.25); ax.set_ylim(-1, 34)
fig.tight_layout(); fig.savefig("results/learning_curves.png")
rows += [("Scripted heuristic (privileged)", 1, f"{heur.mean():.1f}", f"{heur.std():.1f}", f"{heur.mean():.1f}", "-"),
         ("Random policy", 1, f"{rand.mean():.1f}", f"{rand.std():.1f}", f"{rand.mean():.1f}", "-")]
df = pd.DataFrame(rows, columns=["method", "seeds", "mean_deliveries", "episode_std", "per_seed_means", "steps"])
df.to_csv("results/summary.csv", index=False); print(df.to_string(index=False))
