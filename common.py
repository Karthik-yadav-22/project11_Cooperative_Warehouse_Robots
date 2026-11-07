"""Shared helpers: environment construction, evaluation, random baseline."""
import collections
import gymnasium as gym
import numpy as np
import rware  # noqa: F401  (registers the rware-* environments)
from rware.warehouse import RewardType, _LAYER_SHELFS, _LAYER_AGENTS

ENV_ID = "rware-tiny-2ag-v2"  # 2 robots, 1x3 shelf block, request queue = 2 shelves, 500 steps


def make_env(reward="sparse"):
    """reward='sparse'  -> +1 to the robot that delivers a requested shelf (RWARE default)
       reward='staged'  -> +0.5 on delivery at the goal, +0.5 more when the shelf is returned"""
    rt = RewardType.INDIVIDUAL if reward == "sparse" else RewardType.TWO_STAGE
    return gym.make(ENV_ID, reward_type=rt)


N_AUG = 16
STEP = {0: (0, -1), 3: (1, 0), 1: (0, 1), 2: (-1, 0)}   # Direction value -> (dx, dy)
CW = [0, 3, 1, 2]                                         # UP, RIGHT, DOWN, LEFT (clockwise)


class WarehouseEnv:
    """RWARE wrapper with two independent options (both documented in the report).

    augment=True : append 16 robot-frame 'task-assignment' features (what a warehouse management
                   system would broadcast to each robot).  Base RWARE shows only a 3x3 patch and
                   never reveals which shelves are requested:
        stage one-hot (seeking / carrying requested / carrying other)           3
        next BFS step toward the stage target, in the robot frame (fwd,right,back,left)  4
        standing on the stage target, BFS path length / D                        2
        target vectors (forward, right)/D to nearest requested shelf, goal, empty slot  6
        cell ahead blocked                                                       1
    shaping=True : potential-based shaping F = gamma*Phi(s') - Phi(s)  (Ng et al., 1999),
                   which leaves the optimal policy unchanged:
        stage 0  seeking             Phi = -d(robot, nearest requested shelf)/D
        stage 1  carrying requested  Phi = 1.0 - d_BFS(robot, goal)/D   (shelves block a loaded robot)
        stage 2  carrying other      Phi = 0.5 - d_BFS(robot, empty slot)/D
    The environment reward itself stays the sparse +1 per delivery."""

    def __init__(self, shaping=False, augment=False, gamma=0.99):
        self.env, self.shaping, self.augment, self.gamma = make_env("sparse"), shaping, augment, gamma
        self.u = self.env.unwrapped
        self.action_space = self.env.action_space
        self.H, self.W = self.u.grid_size
        self.D = float(self.H + self.W)

    def _bfs(self, sources, blocked):
        shelf = self.u.grid[_LAYER_SHELFS] != 0
        dist = np.full((self.H, self.W), 1e3)
        q = collections.deque()
        for x, y in sources:
            dist[y, x] = 0; q.append((x, y))
        while q:
            x, y = q.popleft()
            for dx, dy in STEP.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and not (blocked and shelf[ny, nx]) \
                        and dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1; q.append((nx, ny))
        return dist

    def _nearest(self, a, cells):
        return min(cells, key=lambda c: abs(a.x - c[0]) + abs(a.y - c[1]))

    def _phi_and_feats(self):
        u = self.u
        req = [(s.x, s.y) for s in u.request_queue]
        carried = {id(a.carrying_shelf) for a in u.agents if a.carrying_shelf is not None}
        occupied = {(s.x, s.y) for s in u.shelfs if id(s) not in carried}
        empty = [c for c in self.home if c not in occupied] or list(self.home)
        goals = [tuple(g) for g in u.goals]
        maps = {}
        phis, feats = [], []
        for a in u.agents:
            carrying = a.carrying_shelf is not None
            has_req = carrying and a.carrying_shelf in u.request_queue
            stage = 1 if has_req else (2 if carrying else 0)
            if stage == 0: key, src, blk = "req", req, False
            elif stage == 1: key, src, blk = "goal", goals, True
            else: key, src, blk = "slot", empty, True
            if key not in maps: maps[key] = self._bfs(src, blk)
            dist = maps[key]
            pos = (a.x, a.y)
            at_target = pos in src
            nb = []
            for k, (dx, dy) in STEP.items():
                nx, ny = a.x + dx, a.y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    nb.append((dist[ny, nx], k))
            best = min(nb) if nb else (1e3, None)
            d_here = 0.0 if at_target else (dist[a.y, a.x] if not blk else best[0] + 1)
            d_here = d_here if d_here < 1e3 else self.D
            base = (0.0, 1.0, 0.5)[stage]
            phis.append(base - d_here / self.D)
            # ---- features
            hi = CW.index(a.dir.value)
            nxt = np.zeros(4, np.float32)
            if not at_target and best[1] is not None and best[0] < 1e3:
                nxt[(CW.index(best[1]) - hi) % 4] = 1.0
            hx, hy = STEP[a.dir.value]; rx, ry = STEP[CW[(hi + 1) % 4]]
            vec = []
            for t in (self._nearest(a, req), self._nearest(a, goals), self._nearest(a, empty)):
                dx, dy = t[0] - a.x, t[1] - a.y
                vec += [(dx * hx + dy * hy) / self.D, (dx * rx + dy * ry) / self.D]
            ax, ay = a.x + hx, a.y + hy
            blocked_ahead = (not (0 <= ax < self.W and 0 <= ay < self.H)) or bool(u.grid[_LAYER_AGENTS, ay, ax]) \
                or (carrying and bool(u.grid[_LAYER_SHELFS, ay, ax]))
            one = np.zeros(3, np.float32); one[stage] = 1.0
            feats.append(np.concatenate([one, nxt, [float(at_target), d_here / self.D], vec,
                                         [float(blocked_ahead)]]).astype(np.float32))
        return np.array(phis), np.array(feats, dtype=np.float32)

    def _out(self, obs, feats):
        o = np.asarray(obs, dtype=np.float32)
        return np.concatenate([o, feats], 1) if self.augment else o

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.home = {(s.x, s.y) for s in self.u.shelfs}
        self._p, feats = self._phi_and_feats()
        return self._out(obs, feats), info

    def step(self, actions):
        obs, r, term, trunc, info = self.env.step(actions)
        p2, feats = self._phi_and_feats()
        r = np.asarray(r, dtype=np.float32)
        if self.shaping:
            r = r + self.gamma * p2 - self._p
        self._p = p2
        return self._out(obs, feats), list(r), term, trunc, info

    def close(self):
        self.env.close()


def make_train_env(reward, augment):
    return WarehouseEnv(shaping=(reward == "shaped"), augment=bool(augment))


def stack(obs):
    return np.asarray(obs, dtype=np.float32)


def run_episode(env, policy_fn, seed):
    """Run one episode on a SPARSE env; return number of deliveries (= total team reward)."""
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        acts = policy_fn(stack(obs))
        obs, r, term, trunc, _ = env.step(list(acts))
        total += float(np.sum(r))
        done = term or trunc
    return total


def random_policy(n_agents=2, n_actions=5, rng=None):
    rng = rng or np.random.default_rng(0)
    return lambda obs: rng.integers(0, n_actions, size=n_agents)


def evaluate(policy_fn, n_episodes=20, seed0=10_000, augment=False):
    env = WarehouseEnv(shaping=False, augment=augment)
    scores = [run_episode(env, policy_fn, seed0 + i) for i in range(n_episodes)]
    env.close()
    return float(np.mean(scores)), float(np.std(scores)), scores
