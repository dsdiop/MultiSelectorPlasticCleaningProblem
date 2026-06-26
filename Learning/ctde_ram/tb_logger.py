"""
Thin TensorBoard logger.

This wrapper is intentionally louder than the old version. If TensorBoard is not
available, it tells you exactly why. If it is available, it exposes the concrete
event directory so you can run:

    tensorboard --logdir <that directory or its parent>

It still degrades to no-op logging when tensorboard is missing, so experiments do
not crash just because visualization is unavailable.
"""
import os
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
    _TB_IMPORT_ERROR = None
except Exception as exc:
    _TB_OK = False
    _TB_IMPORT_ERROR = exc


class TBLogger:
    def __init__(self, log_dir: str, run_name: str = "ctde_ram", verbose: bool = True):
        self.enabled = _TB_OK
        self.episode = 0
        self.global_step = 0
        self.log_dir = os.path.abspath(os.path.join(log_dir, run_name))
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if verbose:
                print(f"[tensorboard] logging to: {self.log_dir}")
        else:
            self.writer = None
            if verbose:
                print(
                    "[tensorboard] disabled: torch.utils.tensorboard could not be imported. "
                    f"Install tensorboard or check your environment. reason={_TB_IMPORT_ERROR!r}"
                )

    def log_step(self, tag, value):
        if self.enabled:
            self.writer.add_scalar(tag, float(value), self.global_step)

    def log_scalar(self, tag, value, step=None):
        if self.enabled:
            self.writer.add_scalar(tag, float(value), self.global_step if step is None else int(step))

    def log_text(self, tag, text, step=0):
        if self.enabled:
            self.writer.add_text(tag, str(text), int(step))

    def log_hparams_text(self, config: dict):
        if not self.enabled:
            return
        lines = ["| key | value |", "|---|---|"]
        for key in sorted(config):
            lines.append(f"| `{key}` | `{config[key]}` |")
        self.writer.add_text("config/args", "\n".join(lines), 0)

    def flush(self):
        if self.enabled:
            self.writer.flush()

    def step(self):
        self.global_step += 1

    def log_episode_metrics(self, metrics, prefix="train"):
        if self.enabled:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.writer.add_scalar(f"{prefix}/{key}", float(value), self.episode)
                elif isinstance(value, (list, tuple)) and value:
                    try:
                        self.writer.add_scalar(f"{prefix}/{key}_mean", float(np.mean(value)), self.episode)
                    except Exception:
                        pass
            self.writer.flush()
        self.episode += 1

    def log_pareto(self, pareto_result, ep=None):
        if not self.enabled:
            return
        e = ep if ep is not None else self.episode
        self.writer.add_scalar("eval/hypervolume", float(pareto_result["hypervolume"]), e)
        for (w, m) in pareto_result["per_weight"]:
            tag_prefix = "eval/w_" + "_".join(f"{v:.2f}" for v in w)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{tag_prefix}/{k}", float(v), e)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            pts = pareto_result["all_points"]
            front = pareto_result["pareto_front"]
            ax.scatter(pts[:, 0], pts[:, 1], c="gray", s=18, label="evaluated")
            ax.scatter(front[:, 0], front[:, 1], c="red", s=40, label="Pareto")
            ax.set_xlabel("coverage"); ax.set_ylabel("trash_cleaned")
            ax.set_title(f"HV={pareto_result['hypervolume']:.3f}"); ax.legend(loc="best")
            self.writer.add_figure("eval/pareto_front", fig, e)
            plt.close(fig)
        except Exception as ex:
            self.writer.add_text("eval/pareto_warning", repr(ex), e)
        self.writer.flush()

    def log_role_histogram(self, role_counts, ep=None):
        if not self.enabled:
            return
        e = ep if ep is not None else self.episode
        for k, c in enumerate(role_counts):
            self.writer.add_scalar(f"eval/role_count_{k}", float(c), e)
        self.writer.flush()

    def close(self):
        if self.enabled:
            self.writer.flush()
            self.writer.close()
