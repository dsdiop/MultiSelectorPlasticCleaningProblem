"""Evaluate the published one-step greedy selector on the Expert_nu substrate."""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from .project_env import build_env_factory_from_existing_project
    from .experiment_io import save_pareto_artifacts, write_json
    from .pareto import hypervolume, pareto_front
except ImportError:
    from project_env import build_env_factory_from_existing_project
    from experiment_io import save_pareto_artifacts, write_json
    from pareto import hypervolume, pareto_front


class GreedyExpertNuSelector:
    """GreedyAgent reward look-ahead adapted to role-ordered CTDE rewards."""

    def __init__(self, env, greedy_type="wpop", reward_stats=None, normalize_rewards=True):
        self.env = env
        self.greedy_type = greedy_type
        self.normalize_rewards = bool(normalize_rewards)
        self.candidates = np.asarray(
            list(itertools.product([0.0, 1.0], repeat=env.N)), dtype=np.float32
        )
        if reward_stats is None:
            self.reward_min = np.zeros(env.K, dtype=np.float32)
            self.reward_max = np.ones(env.K, dtype=np.float32)
        else:
            self.reward_min = np.asarray(reward_stats["min_reward"], dtype=np.float32).copy()
            self.reward_max = np.asarray(reward_stats["max_reward"], dtype=np.float32).copy()

    def _simulate_reward(self, nu, cache):
        active_ids = self.env._active_ids()
        states = self.env._state_float32(active_ids)
        condition = np.asarray(nu, dtype=bool)
        if not self.env.expert_nu.masked_actions:
            actions = self.env.expert_nu.select_action(states, condition=condition)
        else:
            actions = self.env.expert_nu.select_masked_action(
                states=states,
                positions=self.env.env.fleet.get_positions(),
                condition=condition,
            )
        action_dict = {
            agent_id: int(action)
            for agent_id, action in actions.items()
            if agent_id in active_ids
        }
        cache_key = tuple(sorted(action_dict.items()))
        if cache_key not in cache:
            cache[cache_key] = self.env.env.simulate_step(action_dict)
        rewards = self.env._format_rewards(cache[cache_key])
        return np.stack(rewards, axis=0).sum(axis=0)

    def _scalarize(self, rewards, weights):
        if self.greedy_type == "ws":
            return (rewards * weights).sum(axis=1)
        if self.greedy_type == "wp":
            return (rewards ** 3 * weights).sum(axis=1)
        if self.greedy_type == "wpop":
            return np.prod(np.maximum(rewards, 1e-8) ** weights, axis=1)
        if self.greedy_type == "ewc":
            gains = np.exp(weights) - 1.0
            return (gains * np.exp(np.clip(rewards, -20.0, 20.0))).sum(axis=1)
        raise ValueError(f"Unknown greedy type: {self.greedy_type}")

    def select(self, weights):
        cache = {}
        rewards = np.stack(
            [self._simulate_reward(candidate, cache) for candidate in self.candidates], axis=0
        )
        if self.normalize_rewards:
            self.reward_min = np.minimum(self.reward_min, rewards.min(axis=0))
            self.reward_max = np.maximum(self.reward_max, rewards.max(axis=0))
            denom = np.maximum(self.reward_max - self.reward_min, 1e-8)
            rewards = (rewards - self.reward_min) / denom
        scores = self._scalarize(rewards, np.asarray(weights, dtype=np.float32))
        return self.candidates[int(np.argmax(scores))].copy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Greedy Expert_nu preference probe")
    parser.add_argument("--env", choices=["project"], default="project")
    parser.add_argument("--project-control", choices=["expert_nu"], default="expert_nu")
    parser.add_argument("--path-planner-folder", required=True)
    parser.add_argument("--path-planner-root", default=None)
    parser.add_argument("--dqn-ckpt", default=None)
    parser.add_argument("--map-name", default="malaga_port")
    parser.add_argument("--map-csv", default=None)
    parser.add_argument("--initial-positions", default=None)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--T-role", type=int, default=10)
    parser.add_argument("--probe-points", type=int, default=10)
    parser.add_argument("--probe-episodes", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--greedy-type", choices=["ws", "wp", "wpop", "ewc"], default="wpop")
    parser.add_argument("--no-reward-normalization", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=os.path.join("Learning", "ctde_ram", "outputs"))

    # Existing project-environment defaults.
    parser.add_argument("--distance-budget", type=int, default=100)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--no-miopic", dest="miopic", action="store_false")
    parser.set_defaults(miopic=True)
    parser.add_argument("--detection-length", type=int, default=2)
    parser.add_argument("--movement-length", type=int, default=1)
    parser.add_argument("--max-collisions", type=int, default=15)
    parser.add_argument("--reward-type", default="Distance Field")
    parser.add_argument("--convert-to-uint8", action="store_true")
    parser.add_argument("--ground-truth-type", default="macro_plastic")
    parser.add_argument("--number-of-features", type=int, default=1024)
    parser.add_argument("--nettype", default="0")
    parser.add_argument("--archtype", default="v1")
    parser.add_argument("--expert-type", default="ExpertByMapCoverage")
    parser.add_argument("--no-expert-masked-actions", dest="expert_masked_actions", action="store_false")
    parser.add_argument("--no-expert-consensus", dest="expert_consensus", action="store_false")
    parser.set_defaults(expert_masked_actions=True, expert_consensus=True)
    args = parser.parse_args(argv)
    if args.T_role <= 0:
        parser.error("--T-role must be positive")
    if args.probe_points <= 0 or args.probe_episodes <= 0:
        parser.error("probe points and episodes must be positive")
    return args


def collect_reward_stats(env_factory, episodes, seed):
    if episodes <= 0:
        return None
    env = env_factory(0)
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            nu = rng.integers(0, 2, size=env.N).astype(np.float32)
            _, reward_vecs, done, _ = env.step(nu)
            rewards.append(np.stack(reward_vecs, axis=0).sum(axis=0))
    env.close()
    values = np.asarray(rewards, dtype=np.float32)
    return {"min_reward": values.min(axis=0), "max_reward": values.max(axis=0)}


def evaluate_weight(env, selector, weights, episodes, t_role):
    coverages = []
    cleaned = []
    decisions = []
    for _ in range(episodes):
        env.reset()
        done = False
        step = 0
        selected = None
        episode_roles = []
        while not done:
            if selected is None or step % t_role == 0:
                selected = selector.select(weights)
                episode_roles.append(selected.copy())
            _, _, done, _ = env.step(selected)
            step += 1
        coverages.append(env.coverage_pct())
        cleaned.append(env.trash_cleaned_pct())
        decisions.extend(episode_roles)

    role_matrix = np.stack(decisions, axis=0)
    role1_frac = float(role_matrix.mean())
    return {
        "w0": float(weights[0]),
        "w1": float(weights[1]),
        "coverage": float(np.mean(coverages)),
        "trash_cleaned": float(np.mean(cleaned)),
        "n_role_decisions": int(role_matrix.shape[0]),
        "role0_argmax_frac": 1.0 - role1_frac,
        "role0_mean_weight": 1.0 - role1_frac,
        "role1_argmax_frac": role1_frac,
        "role1_mean_weight": role1_frac,
        "mean_nu_role1": role1_frac,
    }


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu" if args.device < 0 or not torch.cuda.is_available() else f"cuda:{args.device}"
    env_factory = build_env_factory_from_existing_project(args)
    reward_stats = collect_reward_stats(env_factory, args.warmup_episodes, args.seed)
    env = env_factory(0)
    selector = GreedyExpertNuSelector(
        env,
        greedy_type=args.greedy_type,
        reward_stats=reward_stats,
        normalize_rewards=not args.no_reward_normalization,
    )

    run_dir = os.path.abspath(os.path.join(args.output_dir, args.run_name))
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "preference_probe.csv")
    progress_path = os.path.join(run_dir, "eval_progress.json")
    write_json(os.path.join(run_dir, "eval_config.json"), {
        "evaluation": vars(args),
        "resolved": {
            "device": device,
            "T_role": 1,
            "selector": args.greedy_type,
            "n_candidates": len(selector.candidates),
            "weight_grid": [
                [float(1.0 - t), float(t)]
                for t in np.linspace(0.0, 1.0, args.probe_points)
            ],
        },
        "heuristic": "one_step_greedy_expert_nu",
        "has_checkpoint": False,
    })

    print("=" * 80)
    print("Greedy Expert_nu probe")
    print("=" * 80)
    print(f"run_name       : {args.run_name}")
    print(f"output         : {csv_path}")
    print(f"device         : {device}")
    print(f"selector       : {args.greedy_type} over {len(selector.candidates)} hard assignments")
    print(f"T_role         : {args.T_role}")
    print(f"probe          : {args.probe_points} weights x {args.probe_episodes} episodes")
    print(f"reward stats   : {reward_stats}")
    print("=" * 80)

    grid = np.linspace(0.0, 1.0, args.probe_points)
    weights = [np.asarray([1.0 - t, t], dtype=np.float32) for t in grid]
    iterator = tqdm(weights, desc="greedy probe", unit="w", dynamic_ncols=True) if tqdm else weights
    rows = []
    write_json(progress_path, {
        "status": "running", "completed": 0, "total": len(weights), "points": []
    })
    for weight in iterator:
        row = evaluate_weight(env, selector, weight, args.probe_episodes, args.T_role)
        rows.append(row)
        write_json(progress_path, {
            "status": "running",
            "completed": len(rows),
            "total": len(weights),
            "points": rows,
        })
    env.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    points = np.asarray(
        [[row["coverage"], row["trash_cleaned"]] for row in rows], dtype=np.float64
    )
    front = pareto_front(points)
    result = {
        "all_points": points,
        "pareto_front": front,
        "hypervolume": hypervolume(front, np.zeros(2, dtype=np.float64)),
        "per_weight": [((row["w0"], row["w1"]), row) for row in rows],
        "ref_point": np.zeros(2, dtype=np.float64),
    }
    artifacts = save_pareto_artifacts(result, run_dir, "pareto_eval")
    write_json(progress_path, {
        "status": "complete",
        "completed": len(rows),
        "total": len(weights),
        "points": rows,
        "hypervolume": float(result["hypervolume"]),
        "artifacts": artifacts,
    })
    write_json(os.path.join(run_dir, "summary.json"), {
        "heuristic": "one_step_greedy_expert_nu",
        "greedy_type": args.greedy_type,
        "hypervolume": float(result["hypervolume"]),
        "front_size": int(front.shape[0]),
        "n_points": int(points.shape[0]),
        "artifacts": artifacts,
        "preference_probe": csv_path,
    })

    for row in rows:
        print(
            f"w=({row['w0']:.2f},{row['w1']:.2f}) "
            f"coverage={row['coverage']:.3f} cleaned={row['trash_cleaned']:.3f} "
            f"role1={row['role1_argmax_frac']:.3f}"
        )
    print(f"[probe] wrote: {csv_path}")
    print(f"[eval] pareto_png: {artifacts['png']}")


if __name__ == "__main__":
    main()
