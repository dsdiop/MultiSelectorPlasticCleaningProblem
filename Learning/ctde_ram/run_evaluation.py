"""Evaluate a saved CTDE-RAM checkpoint.

This script is intentionally separate from training. Point it at any checkpoint
saved by run_experiment.py, including old `latest.pt`, `final.pt`, or
`best_hv.pt`, and it reconstructs the trainer/env from the checkpoint config.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch

try:
    from .experiment_io import args_to_dict, ensure_dir, save_pareto_artifacts, timestamp, write_json
    from .run_experiment import build_env, build_trainer, parse_args as parse_train_args
except ImportError:
    from experiment_io import args_to_dict, ensure_dir, save_pareto_artifacts, timestamp, write_json
    from run_experiment import build_env, build_trainer, parse_args as parse_train_args


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to final.pt, latest.pt, best_hv.pt, etc.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory for eval JSON/CSV/PNG outputs.")
    p.add_argument("--eval-name", type=str, default=None, help="Name for this evaluation folder.")
    p.add_argument("--device", type=int, default=None, help="Override saved device. -1 for CPU.")
    p.add_argument("--episodes-per-w", type=int, default=5)
    p.add_argument("--points", type=int, default=11, help="Number of weights from (1,0) to (0,1).")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help='Optional JSON list of weights, e.g. "[[1,0],[0.5,0.5],[0,1]]".',
    )
    p.add_argument("--probe", action="store_true", help="Also run the preference-sensitivity probe.")
    p.add_argument("--probe-episodes", type=int, default=3)
    return p.parse_args()


def namespace_from_checkpoint(ckpt: dict, eval_args) -> argparse.Namespace:
    # Start from current parser defaults so new code has every expected attr,
    # then overlay the saved training config.
    train_args = parse_train_args([])
    saved = ckpt.get("run_config", {})
    for key, value in saved.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    runtime = ckpt.get("trainer_runtime", {})
    if "film_parameterization" not in saved:
        train_args.film_parameterization = "legacy"
    if getattr(train_args, "T_role", None) is None and runtime.get("T_role") is not None:
        train_args.T_role = int(runtime["T_role"])

    if eval_args.device is not None:
        train_args.device = int(eval_args.device)
    train_args.episodes = 0
    # Evaluation should not create a new training run inside the old output dir.
    train_args.run_name = f"eval_loader_{timestamp()}"
    return train_args


def build_weight_grid(eval_args, k: int):
    if eval_args.weights:
        return [tuple(float(x) for x in row) for row in json.loads(eval_args.weights)]
    if k != 2:
        raise ValueError("--points default grid only supports K=2; pass --weights for K!=2.")
    if eval_args.points <= 1:
        return [(1.0, 0.0)]
    return [(float(1.0 - t), float(t)) for t in np.linspace(0.0, 1.0, int(eval_args.points))]


def main():
    eval_args = parse_args()
    ckpt_path = os.path.abspath(eval_args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = namespace_from_checkpoint(ckpt, eval_args)

    base_dir = eval_args.output_dir
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "offline_eval")
    eval_name = eval_args.eval_name or f"eval_{timestamp()}"
    out_dir = ensure_dir(os.path.join(base_dir, eval_name))
    write_json(os.path.join(out_dir, "loaded_checkpoint.json"), {
        "checkpoint": ckpt_path,
        "checkpoint_episode": ckpt.get("episode"),
        "checkpoint_metrics": ckpt.get("metrics", {}),
    })

    env, low_level_backend, t_role = build_env(train_args)
    device = "cpu" if train_args.device < 0 or not torch.cuda.is_available() else f"cuda:{train_args.device}"
    scal_grid = build_weight_grid(eval_args, env.K)
    write_json(os.path.join(out_dir, "eval_config.json"), {
        "evaluation": args_to_dict(eval_args),
        "resolved": {
            "checkpoint": ckpt_path,
            "checkpoint_episode": ckpt.get("episode"),
            "device": device,
            "low_level_backend": low_level_backend,
            "T_role": t_role,
            "weight_grid": scal_grid,
        },
        "saved_run_config": ckpt.get("run_config", {}),
        "reconstructed_train_config": args_to_dict(train_args),
        "trainer_runtime": ckpt.get("trainer_runtime", {}),
    })
    trainer = build_trainer(
        train_args,
        env,
        low_level_backend,
        t_role,
        device,
        tb_logdir=out_dir,
        tb_runname="tensorboard",
    )
    trainer.load_checkpoint(ckpt_path, load_optimizers=False, map_location=device)
    trainer.tb.log_text("eval/checkpoint", ckpt_path, step=0)

    progress_path = os.path.join(out_dir, "eval_progress.json")
    completed_points = []

    def save_eval_progress(weight, metrics, completed, total):
        completed_points.append({
            "weight": [float(x) for x in weight],
            "coverage": float(metrics["coverage"]),
            "trash_cleaned": float(metrics["trash_cleaned"]),
            "n_switches": float(metrics.get("n_switches", 0.0)),
        })
        write_json(progress_path, {
            "status": "running",
            "checkpoint": ckpt_path,
            "completed": int(completed),
            "total": int(total),
            "points": completed_points,
        })

    write_json(progress_path, {
        "status": "running",
        "checkpoint": ckpt_path,
        "completed": 0,
        "total": len(scal_grid),
        "points": [],
    })
    result = trainer.evaluate_pareto(
        env,
        scal_grid=scal_grid,
        n_episodes_per_w=eval_args.episodes_per_w,
        progress_callback=save_eval_progress,
    )
    paths = save_pareto_artifacts(result, out_dir, "pareto_eval")
    write_json(progress_path, {
        "status": "complete",
        "checkpoint": ckpt_path,
        "completed": len(completed_points),
        "total": len(scal_grid),
        "points": completed_points,
        "hypervolume": float(result["hypervolume"]),
        "artifacts": paths,
    })
    write_json(os.path.join(out_dir, "summary.json"), {
        "checkpoint": ckpt_path,
        "hypervolume": float(result["hypervolume"]),
        "front_size": int(result["pareto_front"].shape[0]),
        "n_points": int(result["all_points"].shape[0]),
        "artifacts": paths,
        "tensorboard_dir": trainer.tb.log_dir,
    })

    if eval_args.probe:
        trainer.probe_preference_sensitivity(
            env,
            scal_grid=scal_grid,
            n_episodes_per_w=eval_args.probe_episodes,
            save_csv=os.path.join(out_dir, "preference_probe.csv"),
            plot_path=os.path.join(out_dir, "preference_probe.png"),
            pareto_plot_path=os.path.join(out_dir, "preference_probe.pareto.png"),
        )

    trainer.close()
    if hasattr(env, "close"):
        env.close()
    print(f"[eval] output_dir={out_dir}")
    print(f"[eval] hypervolume={float(result['hypervolume']):.4f}")
    print(f"[eval] progress_json={progress_path}")
    print(f"[eval] pareto_png={paths['png']}")
    print(f"[tensorboard] view with: tensorboard --logdir \"{os.path.join(out_dir, 'tensorboard')}\"")


if __name__ == "__main__":
    main()
