#!/usr/bin/env python3
"""One-off backfill for missing F4 probe role/Pareto figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Learning.ctde_ram.experiment_io import plot_pareto
from Learning.ctde_ram.pareto import hypervolume, pareto_front


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "Learning" / "ctde_ram" / "outputs"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def role_plot(rows, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [row["w0"] for row in rows]
    ax.plot(x, [row["role0_argmax_frac"] for row in rows], marker="o", label="clean role (0)")
    ax.plot(x, [row["role1_argmax_frac"] for row in rows], marker="o", label="explore role (1)")
    ax.set_xlabel("cleaning preference $w_0$")
    ax.set_ylabel("selected role fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Hard-role preference sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pareto_plot(rows, path: Path):
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
    if not plot_pareto(result, str(path)):
        raise RuntimeError(f"Could not create {path}")


def main():
    probe_csvs = sorted(
        path for path in OUTPUTS.rglob("*probe.csv") if "F4" in str(path)
    )
    if not probe_csvs:
        print("No F4 probe CSV files found")
        return

    created = 0
    for csv_path in probe_csvs:
        rows = load_rows(csv_path)
        role_path = csv_path.with_suffix(".png")
        pareto_path = csv_path.with_suffix(".pareto.png")
        if not role_path.exists():
            role_plot(rows, role_path)
            print(f"created role plot:   {role_path}")
            created += 1
        else:
            print(f"exists role plot:    {role_path}")
        if not pareto_path.exists():
            pareto_plot(rows, pareto_path)
            print(f"created Pareto plot: {pareto_path}")
            created += 1
        else:
            print(f"exists Pareto plot:  {pareto_path}")
    print(f"done: {created} figure(s) created for {len(probe_csvs)} F4 probe(s)")


if __name__ == "__main__":
    main()
