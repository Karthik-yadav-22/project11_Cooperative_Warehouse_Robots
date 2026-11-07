# Cooperative Warehouse Robots

Project 11, Unit 5: Real-World RL Homework Projects

## Problem statement
"Agents learn to avoid collisions and deliver items efficiently (PettingZoo or an RWARE environment)."

## About the project
Two warehouse robots must fetch the shelves customers have requested, carry them to a goal station and put them back, without blocking each other in the narrow aisles. We model this as a cooperative multi-agent problem: each robot acts on its own local observation, all robots share one policy, and the team score is the number of shelves delivered in a 500-step episode.

- **Environment:** RWARE (Robotic Warehouse), `rware-tiny-2ag-v2`: a 10 x 11 grid, 32 shelves, 2 goal cells and a request queue of 2 shelves. The reward is sparse: +1 for each delivery.
- **Algorithms:** Random (baseline), a scripted greedy heuristic that sees the full global state, and IPPO (independent PPO with parameter sharing, one 2 x 128 actor-critic used by both robots).
- **Variants:** sparse or potential-based shaped reward, each with the base 3 x 3 observation or with 16 extra robot-frame task hints (stage, next path step, target vectors, blocked-ahead).
- **Setup:** 1 to 2 million training steps per run, 8 parallel environments, 1 to 2 seeds. Every run is scored on the unshaped sparse reward over 50 held-out evaluation episodes.
- **Metrics:** deliveries per episode, learning curves and sample efficiency.
- **Result:** IPPO with task hints delivers about 27 shelves per episode (26.9 with sparse and with shaped reward), against 15.7 for the scripted heuristic and 0 for Random. Without the hints, nothing learned. Potential-based shaping gave no measurable gain once the hints were present.

| Method | Deliveries per episode |
|---|---|
| Random policy | 0.0 |
| Scripted heuristic (global state) | 15.7 |
| IPPO, base observations (sparse or shaped) | 0.0 |
| IPPO + task hints, sparse reward | 26.9 |
| IPPO + task hints, shaped reward | 26.9 |

## Files
| File | Description |
|---|---|
| `common.py` | Environment wrapper (task hints, potential-based shaping) and evaluation helpers |
| `train_ippo.py` | IPPO training script |
| `baseline_random.py` | Random-policy baseline |
| `baseline_heuristic.py` | Scripted greedy heuristic baseline |
| `plot_results.py` | Learning curves and summary table |
| `requirements.txt` | Python dependencies |
| `results/` | Learning-curve CSVs, final scores, trained weights, `learning_curves.png`, `summary.csv` |
| `Project11_Report.pdf` | Problem formulation, learning curves, code and five-slide presentation |

## How to run
1. Install the dependencies: `pip install -r requirements.txt`
2. Run the baselines: `python baseline_random.py` and `python baseline_heuristic.py`
3. Train, for example the best variant: `python train_ippo.py --reward shaped --aug 1 --seed 0 --steps 1000000`
   - `--reward` is `sparse` or `shaped`; `--aug` is `0` (base observations) or `1` (task hints).
4. Plot the results: `python plot_results.py`, which writes `results/learning_curves.png` and `results/summary.csv`.

Training runs on CPU; one 1M-step run takes roughly 15 to 20 minutes.

## Limitations
- The task hints do the heavy lifting: with the base 3 x 3 view, IPPO delivered 0 shelves in up to 2 million steps.
- Shaping showed no gain (26.9 against 26.9) with 2 seeds each.
- Robots occasionally deadlock, giving some 0-delivery episodes.
- Small and simulated: two robots, one layout, hints that assume a central system knows every request.

## Team
| Name | Roll No. |
|---|---|
| Gonti Karthik Yadav | 100523729022 |
| I Maithreyi | 100523729024 |
| M Sriharshith | 100523729037 |
