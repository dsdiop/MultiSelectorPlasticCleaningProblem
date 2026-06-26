"""
Pareto front + hypervolume utilities (2D maximization), zero deps beyond NumPy.

Note: the Elicit long-message nD hypervolume sweep was self-flagged broken; we keep only
the correct exact 2D routine and raise for D>2 (use pymoo.indicators.hv.HV there).
This matches your greedy paper, which evaluates the (cleaning, exploration) front in 2D.
"""
from typing import List, Tuple
import numpy as np


def is_dominated(p, q) -> bool:
    # maximization: q dominates p iff q >= p everywhere and q > p somewhere
    return bool(np.all(q >= p) and np.any(q > p))


def pareto_front(points: np.ndarray) -> np.ndarray:
    P = points.shape[0]
    keep = np.ones(P, dtype=bool)
    for i in range(P):
        if not keep[i]:
            continue
        for j in range(P):
            if i == j:
                continue
            if is_dominated(points[i], points[j]):
                keep[i] = False
                break
    return points[keep]


def hypervolume_2d(front: np.ndarray, ref: np.ndarray) -> float:
    if front.shape[0] == 0:
        return 0.0
    order = np.argsort(-front[:, 0])
    sorted_front = front[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in sorted_front:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)


def hypervolume(front: np.ndarray, ref: np.ndarray) -> float:
    if front.shape[1] == 2:
        return hypervolume_2d(front, ref)
    raise NotImplementedError("D>2: use pymoo.indicators.hv.HV")


def sweep_scalarizations(evaluator, scal_grid, objective_keys, ref_point) -> dict:
    # evaluator(w: np.ndarray) -> dict with objective_keys
    points, raw = [], []
    for w in scal_grid:
        m = evaluator(np.array(w, dtype=np.float32))
        raw.append((tuple(w), m))
        points.append([m[k] for k in objective_keys])
    points = np.array(points, dtype=np.float64)
    front = pareto_front(points)
    hv = hypervolume(front, np.array(ref_point, dtype=np.float64))
    return {
        "all_points": points,
        "pareto_front": front,
        "hypervolume": hv,
        "per_weight": raw,
        "ref_point": np.array(ref_point, dtype=np.float64),
    }
