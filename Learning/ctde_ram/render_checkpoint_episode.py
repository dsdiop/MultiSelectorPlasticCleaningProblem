"""Render one deterministic CTDE-RAM episode from a saved experiment.

The six panels mirror ``MultiAgentPatrolling.render`` and add fleet-level
information that is useful when inspecting a role policy: agent IDs, current
role assignments, and the selected preference.

Example
-------
python Learning/ctde_ram/render_checkpoint_episode.py \
  --experiment Learning/ctde_ram/outputs/my_run \
  --checkpoint latest.pt \
  --preference 0.7 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .run_experiment import build_env, build_trainer, parse_args as parse_train_args
except ImportError:
    from run_experiment import build_env, build_trainer, parse_args as parse_train_args


ROLE_NAMES = {0: "clean", 1: "explore"}
ROLE_SHORT = {0: "C", 1: "E"}
ROLE_COLORS = {0: "deepskyblue", 1: "darkorange"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one episode using an experiment config and CTDE-RAM checkpoint."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint path, or a filename such as latest.pt resolved under checkpoints/.",
    )
    parser.add_argument(
        "--preference", nargs=2, type=float, required=True, metavar=("CLEAN", "COVERAGE"),
        help="Mission preference weights; they are normalized to sum to one.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Episode seed; defaults to config seed.")
    parser.add_argument("--device", type=int, default=None, help="Device override: -1 CPU, 0 GPU 0, etc.")
    parser.add_argument("--pause", type=float, default=0.03, help="Seconds between rendered steps.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional early stop for debugging.")
    parser.add_argument("--save", type=str, default=None, help="Final-frame PNG path.")
    parser.add_argument("--no-show", action="store_true", help="Use a headless backend and do not open a window.")
    parser.add_argument("--no-block", action="store_true", help="Close instead of blocking after the episode.")
    return parser.parse_args()


def resolve_inputs(cli: argparse.Namespace) -> tuple[Path, Path, Path]:
    experiment = Path(cli.experiment).expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment}")
    config_path = experiment / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    checkpoint = Path(cli.checkpoint).expanduser()
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        checkpoint = experiment / "checkpoints" / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    save_path = Path(cli.save).expanduser() if cli.save else experiment / "rendered_episode.png"
    save_path = save_path.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return experiment, checkpoint, save_path


def reconstruct_train_args(config_path: Path, cli: argparse.Namespace) -> argparse.Namespace:
    with config_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    train_args = parse_train_args([])
    for key, value in saved.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    # Old checkpoints predate the bounded FiLM parameterization.
    if "film_parameterization" not in saved:
        train_args.film_parameterization = "legacy"
    if cli.device is not None:
        train_args.device = int(cli.device)
    if cli.seed is not None:
        train_args.seed = int(cli.seed)
    train_args.episodes = 0
    return train_args


class EpisodeRenderer:
    """Six-channel renderer compatible with interactive and Agg backends."""

    def __init__(self, env, preference: np.ndarray, pause: float, interactive: bool):
        import matplotlib.pyplot as plt
        from Environment.PatrollingEnvironments import background_colormap, macroplastic_colormap

        self.plt = plt
        self.raw = getattr(env, "env", env)
        self.preference = preference
        self.pause = max(float(pause), 0.0)
        self.interactive = bool(interactive)
        self.background_colormap = background_colormap
        self.macroplastic_colormap = macroplastic_colormap
        self.fig, self.axes = plt.subplots(1, 6, figsize=(19, 4.8), constrained_layout=True)
        self.images = []
        self.agent_artists = []
        if self.interactive:
            plt.ion()

    def _active_focal_agent(self) -> int:
        active = getattr(self.raw, "active_agents", None)
        if active is None:
            return 0
        if isinstance(active, dict):
            ids = [int(i) for i, enabled in active.items() if enabled]
        else:
            ids = list(np.flatnonzero(np.asarray(active, dtype=bool)))
        return ids[0] if ids else 0

    def _channels(self):
        raw = self.raw
        scenario = np.asarray(raw.scenario_map)
        visitable = np.asarray(raw.visitable_locations, dtype=int)
        focal = self._active_focal_agent()
        state = getattr(raw, "state", {})
        focal_state = np.asarray(state[focal])

        real_gt = np.full(scenario.shape, np.nan, dtype=np.float32)
        gt = np.asarray(getattr(raw, "macro_plastic_gt", raw.gt.read()))
        real_gt[visitable[:, 0], visitable[:, 1]] = gt[visitable[:, 0], visitable[:, 1]]

        model = np.full(scenario.shape, np.nan, dtype=np.float32)
        model[visitable[:, 0], visitable[:, 1]] = focal_state[0, visitable[:, 0], visitable[:, 1]]
        return [
            scenario,
            real_gt,
            model,
            focal_state[1],
            focal_state[2],
            np.asarray(raw.fleet.historic_visited_mask),
        ], focal

    def _draw_agents(self, roles: np.ndarray):
        for artist in self.agent_artists:
            artist.remove()
        self.agent_artists.clear()
        positions = np.asarray(self.raw.fleet.get_positions())
        for agent_id, position in enumerate(positions):
            row, col = float(position[0]), float(position[1])
            role = int(roles[agent_id])
            color = ROLE_COLORS.get(role, "magenta")
            marker = self.axes[0].scatter(
                [col], [row], s=180, c=color, edgecolors="black", linewidths=1.5, zorder=5
            )
            label = self.axes[0].text(
                col, row, f"A{agent_id}\n{ROLE_SHORT.get(role, '?')}",
                ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=6,
            )
            self.agent_artists.extend([marker, label])

    def update(self, roles: np.ndarray, step: int, coverage: float, clean: float):
        channels, focal = self._channels()
        titles = [
            "Navigation map + agents",
            "Real importance GT",
            f"Model (agent {focal})",
            f"Agent {focal} position",
            f"Others than agent {focal}",
            "Redundancy / visited mask",
        ]
        cmaps = [self.background_colormap, self.macroplastic_colormap,
                 self.macroplastic_colormap, "gray", "gray", "gray"]
        if not self.images:
            for axis, channel, title, cmap in zip(self.axes, channels, titles, cmaps):
                image = axis.imshow(channel, cmap=cmap, interpolation="nearest")
                axis.set_title(title, fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
                self.images.append(image)
        else:
            for image, channel in zip(self.images, channels):
                image.set_data(channel)
                finite = np.asarray(channel)[np.isfinite(channel)]
                if finite.size and image.get_cmap().name != self.background_colormap.name:
                    lo, hi = float(finite.min()), float(finite.max())
                    image.set_clim(lo, hi if hi > lo else lo + 1e-6)
            self.axes[2].set_title(f"Model (agent {focal})", fontsize=10)
            self.axes[3].set_title(f"Agent {focal} position", fontsize=10)
            self.axes[4].set_title(f"Others than agent {focal}", fontsize=10)

        self._draw_agents(roles)
        role_text = "  ".join(
            f"A{i}:{ROLE_NAMES.get(int(role), str(int(role)))}" for i, role in enumerate(roles)
        )
        self.fig.suptitle(
            f"preference clean={self.preference[0]:.3f}, coverage={self.preference[1]:.3f}  |  "
            f"step={step}  clean={clean:.3f}  coverage={coverage:.3f}\n{role_text}",
            fontsize=12,
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.interactive and self.pause > 0:
            self.plt.pause(self.pause)

    def finish(self, save_path: Path, block: bool):
        self.fig.savefig(save_path, dpi=160, bbox_inches="tight")
        if self.interactive and block:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.close(self.fig)


def run_episode(trainer, env, preference: np.ndarray, renderer: EpisodeRenderer,
                seed: int, max_steps: int | None) -> dict:
    trainer._active_env = env
    obs_all = env.reset(seed=seed)
    previous_roles = np.zeros(trainer.N, dtype=np.int64)
    done = False
    step = 0
    decisions = 0
    roles_np = previous_roles.copy()

    while not done and (max_steps is None or step < max_steps):
        roles, _ = trainer._select_main_hard_roles(
            obs_all, previous_roles, preference, epsilon=0.0, training=False
        )
        roles_np = roles.detach().cpu().numpy().astype(np.int64)
        renderer.update(roles_np, step, env.coverage_pct(), env.trash_cleaned_pct())
        duration = 0
        while duration < trainer.T_role and not done:
            if max_steps is not None and step >= max_steps:
                break
            obs_all, _, done, _, _ = trainer._step_env_with_hard_roles(
                env, obs_all, roles, epsilon_low=0.0
            )
            duration += 1
            step += 1
            renderer.update(roles_np, step, env.coverage_pct(), env.trash_cleaned_pct())
        previous_roles = roles_np
        decisions += 1

    return {
        "steps": step,
        "role_decisions": decisions,
        "done": bool(done),
        "coverage": float(env.coverage_pct()),
        "trash_cleaned": float(env.trash_cleaned_pct()),
        "last_roles": roles_np.tolist(),
    }


def main() -> None:
    cli = parse_args()
    experiment, checkpoint, save_path = resolve_inputs(cli)
    preference = np.asarray(cli.preference, dtype=np.float32)
    if not np.isfinite(preference).all() or np.any(preference < 0) or float(preference.sum()) <= 0:
        raise ValueError("--preference must contain finite, non-negative weights with a positive sum")
    preference /= preference.sum()

    if cli.no_show:
        import matplotlib
        matplotlib.use("Agg", force=True)

    train_args = reconstruct_train_args(experiment / "config.json", cli)
    device = (
        "cpu" if int(train_args.device) < 0 or not torch.cuda.is_available()
        else f"cuda:{int(train_args.device)}"
    )
    env = trainer = None
    with tempfile.TemporaryDirectory(prefix="ctde_render_") as temp_logdir:
        try:
            env, low_level_backend, t_role = build_env(train_args)
            trainer = build_trainer(
                train_args, env, low_level_backend, t_role, device,
                tb_logdir=temp_logdir, tb_runname="tensorboard",
            )
            trainer.load_checkpoint(str(checkpoint), load_optimizers=False, map_location=device)
            renderer = EpisodeRenderer(env, preference, cli.pause, interactive=not cli.no_show)
            result = run_episode(
                trainer, env, preference, renderer,
                seed=int(train_args.seed), max_steps=cli.max_steps,
            )
            renderer.finish(save_path, block=not cli.no_block and not cli.no_show)
        finally:
            if trainer is not None:
                trainer.close()
            if env is not None and hasattr(env, "close"):
                env.close()

    print(f"[render] experiment={experiment}")
    print(f"[render] checkpoint={checkpoint}")
    print(f"[render] device={device} preference={preference.tolist()}")
    print(f"[render] result={json.dumps(result, sort_keys=True)}")
    print(f"[render] final_frame={save_path}")


if __name__ == "__main__":
    main()
