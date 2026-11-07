"""Random-policy baseline on the same 50 evaluation seeds used for the trained policies."""
import numpy as np
from common import evaluate, random_policy

m, sd, scores = evaluate(random_policy(rng=np.random.default_rng(0)), 50, seed0=90_000)
np.save("results/random_scores.npy", scores)
print(f"random baseline: {m:.3f} ± {sd:.3f} deliveries/episode over 50 episodes")
