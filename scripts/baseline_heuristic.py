"""Scripted greedy heuristic (uses privileged global state; a strong reference, not a learner).

Each robot: (1) walk to the nearest *unclaimed* requested shelf and pick it up,
(2) BFS-route to the nearest goal (loaded robots cannot cross shelves),
(3) after delivery, carry the shelf back to the nearest empty slot and drop it, repeat."""
import collections, numpy as np
from rware.warehouse import Direction, _LAYER_SHELFS, _LAYER_AGENTS
from common import make_env, stack, run_episode

NOOP, FWD, LEFT, RIGHT, TOGGLE = range(5)
CW = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
STEP = {Direction.UP: (0, -1), Direction.DOWN: (0, 1), Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0)}


class Heuristic:
    def __init__(self, env):
        self.u = env.unwrapped
        self.home = None

    def _next_cell(self, a, target, loaded):
        u = self.u; H, W = u.grid_size
        shelf = u.grid[_LAYER_SHELFS]; ag = u.grid[_LAYER_AGENTS]
        prev = {(a.x, a.y): None}; q = collections.deque([(a.x, a.y)])
        while q:
            c = q.popleft()
            if c == target: break
            for d in STEP.values():
                n = (c[0] + d[0], c[1] + d[1])
                if not (0 <= n[0] < W and 0 <= n[1] < H) or n in prev: continue
                if loaded and shelf[n[1], n[0]]: continue
                if ag[n[1], n[0]] and n != target: continue          # other robot blocks
                prev[n] = c; q.append(n)
        if target not in prev: return None
        c = target
        while prev[c] != (a.x, a.y) and prev[c] is not None: c = prev[c]
        return c

    def _move(self, a, cell):
        if cell is None: return NOOP
        dx, dy = cell[0] - a.x, cell[1] - a.y
        want = [k for k, v in STEP.items() if v == (dx, dy)][0]
        if a.dir == want: return FWD
        diff = (CW.index(want) - CW.index(a.dir)) % 4
        return RIGHT if diff in (1, 2) else LEFT

    def __call__(self, _obs=None):
        u = self.u
        if self.home is None: self.home = {(s.x, s.y) for s in u.shelfs}
        req = [s for s in u.request_queue]
        carried = {id(a.carrying_shelf) for a in u.agents if a.carrying_shelf is not None}
        occupied = {(s.x, s.y) for s in u.shelfs if id(s) not in carried}
        empty = [c for c in self.home if c not in occupied]
        claimed, acts = set(), []
        for a in u.agents:
            pos = (a.x, a.y)
            if a.carrying_shelf is not None and a.carrying_shelf in req:
                tgt = min(u.goals, key=lambda g: abs(g[0]-a.x)+abs(g[1]-a.y))
                acts.append(NOOP if pos == tgt else self._move(a, self._next_cell(a, tgt, True)))
            elif a.carrying_shelf is not None:
                if pos in empty and pos in self.home: acts.append(TOGGLE); continue
                tgt = min(empty, key=lambda c: abs(c[0]-a.x)+abs(c[1]-a.y)) if empty else pos
                acts.append(self._move(a, self._next_cell(a, tgt, True)))
            else:
                cands = [s for s in req if id(s) not in claimed]
                if not cands: acts.append(NOOP); continue
                s = min(cands, key=lambda s: abs(s.x-a.x)+abs(s.y-a.y)); claimed.add(id(s))
                acts.append(TOGGLE if pos == (s.x, s.y) else self._move(a, self._next_cell(a, (s.x, s.y), False)))
        return acts


if __name__ == "__main__":
    env = make_env("sparse"); h = Heuristic(env)
    scores = []
    for i in range(50):
        h.home = None
        scores.append(run_episode(env, lambda o: h(o), 90_000 + i))
    np.save("results/heuristic_scores.npy", scores)
    print(f"heuristic: {np.mean(scores):.2f} ± {np.std(scores):.2f} deliveries/episode (50 episodes)")
