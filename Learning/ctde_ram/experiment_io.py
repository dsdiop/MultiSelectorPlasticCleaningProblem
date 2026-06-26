"""Experiment IO helpers for CTDE-RAM runs.

Everything in here is deliberately boring and explicit: one run directory, JSON
configs, CSV metrics, checkpoint files, and PNG figures. That makes it possible
to train today, come back later, and know exactly which policy produced which
front.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Iterable

import numpy as np


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def make_run_name(args) -> str:
    if getattr(args, "run_name", None):
        return str(args.run_name)
    parts = [
        "ctde_ram",
        str(getattr(args, "env", "env")),
        str(getattr(args, "ram_mode", "ram")),
        str(getattr(args, "soft_ram_arch", "arch")),
        str(getattr(args, "role_scalarization", "ws")),
        str(getattr(args, "q_scalarization", "ws")),
        timestamp(),
    ]
    return "_".join(parts)


def resolve_run_dir(output_dir: str, run_name: str) -> str:
    base_dir = os.path.abspath(os.path.join(output_dir, run_name))
    if not os.path.exists(base_dir):
        return ensure_dir(base_dir)

    suffix = 1
    while True:
        candidate_dir = f"{base_dir}_{suffix}"
        if not os.path.exists(candidate_dir):
            return ensure_dir(candidate_dir)
        suffix += 1


def build_probe_artifact_path(output_dir: str, run_name: str, episodes: int, ext: str) -> str:
    stem = f"{run_name}_episodes_{int(episodes)}_probe"
    return os.path.join(output_dir, f"{stem}.{ext}")


def args_to_dict(args) -> dict:
    return {k: to_jsonable(v) for k, v in vars(args).items()}


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
            out[key] = to_jsonable(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                out[f"{key}_count"] = 0
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                out[f"{key}_mean"] = float(arr.mean())
                out[f"{key}_min"] = float(arr.min())
                out[f"{key}_max"] = float(arr.max())
                out[f"{key}_count"] = int(arr.size)
            else:
                out[key] = json.dumps(to_jsonable(value))
        elif value is None:
            out[key] = ""
        else:
            out[key] = json.dumps(to_jsonable(value))
    return out


def append_csv_row(path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    flat = flatten_metrics(row)
    exists = os.path.exists(path)
    old_rows = []
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            old_rows = list(reader)
    else:
        fieldnames = []

    merged_fields = list(fieldnames)
    for key in flat:
        if key not in merged_fields:
            merged_fields.append(key)

    mode = "w" if fieldnames != merged_fields else ("a" if exists else "w")
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        if mode == "w":
            writer.writeheader()
            for old in old_rows:
                writer.writerow(old)
        writer.writerow(flat)


def write_csv_rows(path: str, rows: Iterable[dict]) -> None:
    rows = [flatten_metrics(r) for r in rows]
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pareto_rows(result: dict) -> list[dict]:
    rows = []
    for w, metrics in result["per_weight"]:
        row = {f"w{i}": float(v) for i, v in enumerate(w)}
        for key, value in metrics.items():
            if key.startswith("_"):
                continue
            row[key] = value
        rows.append(row)
    return rows


def save_pareto_artifacts(result: dict, out_dir: str, prefix: str) -> dict:
    ensure_dir(out_dir)
    paths = {
        "json": os.path.join(out_dir, f"{prefix}.json"),
        "csv": os.path.join(out_dir, f"{prefix}.csv"),
        "png": os.path.join(out_dir, f"{prefix}.png"),
    }
    write_json(paths["json"], result)
    write_csv_rows(paths["csv"], pareto_rows(result))
    if not plot_pareto(result, paths["png"]):
        paths["png"] = None
    return paths


def plot_pareto(result: dict, path: str) -> bool:
    try:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped pareto plot: {exc!r}")
        return False

    pts = np.asarray(result["all_points"], dtype=float)
    front = np.asarray(result["pareto_front"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], c="0.55", s=28, label="evaluated")
    if front.size:
        ax.scatter(front[:, 0], front[:, 1], c="tab:red", s=48, label="Pareto")
    for w, metrics in result["per_weight"]:
        x = float(metrics.get("coverage", np.nan))
        y = float(metrics.get("trash_cleaned", np.nan))
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(",".join(f"{v:.2f}" for v in w), (x, y), fontsize=8)
    ax.set_xlabel("coverage")
    ax.set_ylabel("trash_cleaned")
    ax.set_title(f"HV={float(result['hypervolume']):.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True
