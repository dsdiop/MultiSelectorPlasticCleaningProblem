"""
AquaticMAEnv: toy discretized aquatic env for a homogeneous fleet of N ASVs.

This is the SELF-CONTAINED toy from the Elicit proposal so the whole stack compiles and
trains standalone. It is NOT your MultiAgentPatrolling env. For the paper you swap this
for your real env (see POINTS OF FAILURE in the README): the trainer only needs
reset() -> obs_list, step(actions) -> (obs_list, r_vecs, done, info), and the three
helpers coverage_pct/trash_density/budget_frac, plus attributes obs_dim, K, A.

Reward channels: r_vec[i] = [explore, clean] per agent (K=2), matching your 2-objective
(exploration, cleaning) setup.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch


class AquaticMAEnv:
    def __init__(self, N=4, H=20, W=20, patch_radius=3, max_steps=200,
                 trash_init_cells=30, trash_max=5.0, trash_pickup=1.0,
                 drift_rate=0.0, seed: Optional[int] = None):
        self.N, self.H, self.W = N, H, W
        self.patch_radius, self.max_steps = patch_radius, max_steps
        self.trash_init_cells, self.trash_max = trash_init_cells, trash_max
        self.trash_pickup, self.drift_rate = trash_pickup, drift_rate
        self.A = 5
        self._dyx = [(0, 0), (-1, 0), (0, +1), (+1, 0), (0, -1)]
        self.K = 2
        self.rng = np.random.default_rng(seed)
        self.trash = np.zeros((H, W), dtype=np.float32)
        self.visited = np.zeros((H, W), dtype=np.float32)
        self.positions: List[Tuple[int, int]] = []
        self.step_count = 0
        self._total_trash_collected = 0.0
        self.OUT_OF_BOUNDS = -1.0
        side = 2 * patch_radius + 1
        # trash patch + visited patch + own(row,col,step_frac) + (N-1)*(dy,dx,occ)
        self.obs_dim = side * side * 2 + 3 + (N - 1) * 3

    def reset(self) -> List[np.ndarray]:
        self.trash[:] = 0.0
        self.visited[:] = 0.0
        self.step_count = 0
        self._total_trash_collected = 0.0
        idx = self.rng.choice(self.H * self.W, size=self.trash_init_cells, replace=False)
        for flat in idx:
            r, c = divmod(int(flat), self.W)
            self.trash[r, c] = float(self.rng.uniform(0.5, self.trash_max))
        self._initial_total_trash = float(self.trash.sum())
        pos_idx = self.rng.choice(self.H * self.W, size=self.N, replace=False)
        self.positions = [divmod(int(p), self.W) for p in pos_idx]
        for (r, c) in self.positions:
            self.visited[r, c] = 1.0
        return [self._observation(i) for i in range(self.N)]

    def step(self, actions):
        assert len(actions) == self.N
        r_vecs = [torch.zeros(self.K, dtype=torch.float32) for _ in range(self.N)]
        new_positions = []
        for i, a in enumerate(actions):
            dy, dx = self._dyx[int(a)]
            r, c = self.positions[i]
            nr = int(np.clip(r + dy, 0, self.H - 1))
            nc = int(np.clip(c + dx, 0, self.W - 1))
            new_positions.append((nr, nc))
            if self.visited[nr, nc] == 0.0:
                r_vecs[i][0] = 1.0           # explore reward
                self.visited[nr, nc] = 1.0
            if self.trash[nr, nc] > 0.0:
                picked = min(self.trash_pickup, float(self.trash[nr, nc]))
                self.trash[nr, nc] -= picked
                r_vecs[i][1] = float(picked)  # clean reward
                self._total_trash_collected += picked
        self.positions = new_positions
        if self.drift_rate > 0.0:
            self._drift()
        self.step_count += 1
        done = (self.step_count >= self.max_steps or float(self.trash.sum()) <= 1e-6)
        info = {
            "trash_cleaned": self._total_trash_collected,
            "trash_left": float(self.trash.sum()),
            "coverage_pct": self.coverage_pct(),
        }
        obs_all = [self._observation(i) for i in range(self.N)]
        return obs_all, r_vecs, done, info

    def coverage_pct(self) -> float:
        return float(self.visited.sum() / (self.H * self.W))

    def trash_density(self) -> float:
        return float((self.trash > 0).sum() / (self.H * self.W))

    def budget_frac(self) -> float:
        return float(max(0, self.max_steps - self.step_count) / self.max_steps)

    def _observation(self, i: int) -> np.ndarray:
        r, c = self.positions[i]
        rad = self.patch_radius
        trash_patch = self._extract_patch(self.trash, r, c, rad, self.OUT_OF_BOUNDS)
        visited_patch = self._extract_patch(self.visited, r, c, rad, self.OUT_OF_BOUNDS)
        own = np.array([
            r / max(self.H - 1, 1),
            c / max(self.W - 1, 1),
            self.step_count / max(self.max_steps, 1),
        ], dtype=np.float32)
        teammates = []
        for j in range(self.N):
            if j == i:
                continue
            rj, cj = self.positions[j]
            teammates.append([
                (rj - r) / max(self.H - 1, 1),
                (cj - c) / max(self.W - 1, 1),
                1.0 if (rj, cj) == (r, c) else 0.0,
            ])
        teammates = np.array(teammates, dtype=np.float32).flatten() if teammates else np.zeros(0, dtype=np.float32)
        obs = np.concatenate([
            trash_patch.flatten() / max(self.trash_max, 1.0),
            visited_patch.flatten(),
            own,
            teammates,
        ]).astype(np.float32)
        assert obs.shape[0] == self.obs_dim, (obs.shape, self.obs_dim)
        return obs

    def _extract_patch(self, grid, r, c, rad, pad_value=0.0):
        side = 2 * rad + 1
        patch = np.full((side, side), pad_value, dtype=np.float32)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                rr, cc = r + dy, c + dx
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    patch[dy + rad, dx + rad] = grid[rr, cc]
        return patch

    def _drift(self):
        new = self.trash.copy()
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(self.trash, shift=d, axis=(0, 1))
            if d[0] == -1:
                shifted[-1, :] = 0.0
            elif d[0] == 1:
                shifted[0, :] = 0.0
            elif d[1] == -1:
                shifted[:, -1] = 0.0
            elif d[1] == 1:
                shifted[:, 0] = 0.0
            new += self.drift_rate * (shifted - self.trash)
        self.trash = np.maximum(new, 0.0)

    def render(self) -> str:
        grid = np.full((self.H, self.W), ".", dtype="<U1")
        grid[self.trash > 0.0] = "#"
        for i, (r, c) in enumerate(self.positions):
            grid[r, c] = str(i % 10)
        return "\n".join("".join(row) for row in grid)
