#!/usr/bin/env python3
"""Rank all Malaga-port F4 experiments from their saved evaluation artifacts.

Ranking is lexicographic (descending):
  1. Hypervolume recomputed from the final preference-probe CSV.
  2. Best historical Hypervolume in metrics/eval_summary.csv or eval JSON files.
  3. Highest trash_cleaned value reached in any probe/evaluation point.

Missing values rank below available values. No third-party table package is
required, so this can be run directly in the training environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


PREFIX = "malaga_port_F4_"


@dataclass
class Result:
    directory: Path
    probe_hv: float | None
    historical_hv: float | None
    max_cleaned: float | None
    probe_points: int
    eval_points: int
    probe_file: Path | None


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = list(dict.fromkeys(points))
    return [
        point for i, point in enumerate(unique)
        if not any(
            j != i
            and other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for j, other in enumerate(unique)
        )
    ]


def hypervolume_zero_ref(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    front = sorted(pareto_front(points), key=lambda point: point[0], reverse=True)
    hv = 0.0
    previous_y = 0.0
    for x, y in front:
        if y > previous_y:
            hv += max(x, 0.0) * (y - previous_y)
            previous_y = y
    return hv


def read_probe(directory: Path) -> tuple[float | None, list[float], int, Path | None]:
    files = sorted(directory.glob("*probe.csv"), key=lambda path: (path.stat().st_mtime, path.name))
    if not files:
        return None, [], 0, None
    probe_file = files[-1]
    points: list[tuple[float, float]] = []
    cleaned: list[float] = []
    try:
        with probe_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                coverage = finite(row.get("coverage"))
                trash = finite(row.get("trash_cleaned"))
                if coverage is None or trash is None:
                    continue
                points.append((coverage, trash))
                cleaned.append(trash)
    except (OSError, csv.Error):
        return None, [], 0, probe_file
    return hypervolume_zero_ref(points), cleaned, len(points), probe_file


def read_evaluations(directory: Path) -> tuple[float | None, list[float], int]:
    historical_hv: list[float] = []
    cleaned: list[float] = []
    point_count = 0

    summary = directory / "metrics" / "eval_summary.csv"
    if summary.is_file():
        try:
            with summary.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = finite(row.get("hypervolume"))
                    if value is not None:
                        historical_hv.append(value)
        except (OSError, csv.Error):
            pass

    # JSON files provide the clean extrema and are also a fallback for HV.
    for path in sorted((directory / "eval").glob("eval_ep_*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        value = finite(payload.get("hypervolume"))
        if value is not None:
            historical_hv.append(value)
        for point in payload.get("all_points", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            trash = finite(point[1])
            if trash is not None:
                cleaned.append(trash)
                point_count += 1

    return (max(historical_hv) if historical_hv else None), cleaned, point_count


def inspect(directory: Path) -> Result:
    probe_hv, probe_cleaned, probe_points, probe_file = read_probe(directory)
    historical_hv, eval_cleaned, eval_points = read_evaluations(directory)
    all_cleaned = probe_cleaned + eval_cleaned
    return Result(
        directory=directory,
        probe_hv=probe_hv,
        historical_hv=historical_hv,
        max_cleaned=max(all_cleaned) if all_cleaned else None,
        probe_points=probe_points,
        eval_points=eval_points,
        probe_file=probe_file,
    )


def rank_value(value: float | None) -> float:
    return value if value is not None else float("-inf")


def display_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def shortened(name: str, width: int) -> str:
    if len(name) <= width:
        return name
    if width <= 3:
        return name[:width]
    return name[: width - 1] + "…"


def print_table(results: list[Result], full_names: bool) -> None:
    terminal_width = shutil.get_terminal_size((160, 24)).columns
    fixed_width = 5 + 10 + 10 + 10 + 8 + 5 * 3
    name_width = max(30, terminal_width - fixed_width)
    if full_names:
        name_width = max(name_width, max((len(row.directory.name) for row in results), default=30))
    header = (
        f"{'Rank':>4}  {'Experiment':<{name_width}}  {'Probe HV':>10}  "
        f"{'Best HV':>10}  {'Max clean':>10}  {'P/E pts':>9}"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(results, 1):
        name = row.directory.name
        if name.startswith(PREFIX):
            name = name[len(PREFIX):]
        name = shortened(name, name_width)
        counts = f"{row.probe_points}/{row.eval_points}"
        print(
            f"{index:>4}  {name:<{name_width}}  {display_number(row.probe_hv):>10}  "
            f"{display_number(row.historical_hv):>10}  {display_number(row.max_cleaned):>10}  "
            f"{counts:>9}"
        )
    print("\nRanking: Probe HV ↓, then best historical HV ↓, then maximum trash_cleaned ↓")
    print("HV uses the exact 2D maximization front with reference point (0, 0).")
    print("P/E pts = valid probe points / saved evaluation points; — means unavailable.")


def write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "experiment", "probe_hv", "best_historical_hv",
            "max_trash_cleaned", "probe_points", "eval_points", "probe_file",
        ])
        writer.writeheader()
        for index, row in enumerate(results, 1):
            writer.writerow({
                "rank": index,
                "experiment": row.directory.name,
                "probe_hv": row.probe_hv,
                "best_historical_hv": row.historical_hv,
                "max_trash_cleaned": row.max_cleaned,
                "probe_points": row.probe_points,
                "eval_points": row.eval_points,
                "probe_file": str(row.probe_file) if row.probe_file else "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs", type=Path, default=Path("Learning/ctde_ram/outputs"),
        help="Outputs root (default: Learning/ctde_ram/outputs).",
    )
    parser.add_argument("--full-names", action="store_true", help="Do not truncate experiment names.")
    parser.add_argument("--csv", type=Path, default=None, help="Optionally save the ranked table as CSV.")
    args = parser.parse_args()

    outputs = args.outputs.expanduser().resolve()
    if not outputs.is_dir():
        raise SystemExit(f"Outputs directory does not exist: {outputs}")
    directories = sorted(
        path for path in outputs.iterdir()
        if path.is_dir() and path.name.startswith(PREFIX)
    )
    if not directories:
        raise SystemExit(f"No {PREFIX}* experiments found under {outputs}")

    results = [inspect(directory) for directory in directories]
    results.sort(key=lambda row: (
        rank_value(row.probe_hv),
        rank_value(row.historical_hv),
        rank_value(row.max_cleaned),
        row.directory.name,
    ), reverse=True)
    print_table(results, args.full_names)
    if args.csv:
        output_path = args.csv.expanduser().resolve()
        write_csv(output_path, results)
        print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
