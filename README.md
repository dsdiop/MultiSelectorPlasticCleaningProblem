# MultiSelector Plastic Cleaning Problem

This repository studies cooperative multi-agent cleaning of marine plastic pollution in a map-based patrolling environment. The main novelty is a hierarchical learning framework based on CTDE-RAM: Centralized Training, Decentralized Execution with a Role Assignment Module.

CTDE-RAM learns a high-level selector that assigns role/preference weights to the fleet over time, while a low-level controller executes the resulting behaviors. This makes the system capable of balancing cleaning and coverage in a preference-conditioned way, which is the central contribution of this work.

## What is novel here?

- A role-based hierarchical controller for multi-agent cleaning.
- A learned high-level selector that chooses soft role weights instead of fixed hard roles.
- Preference-conditioned behavior over mission objectives such as cleaning gain and coverage gain.
- Support for multiple RAM variants, including hard baselines and soft CTDE-RAM variants.

## Repository structure

- Environment/: environment definitions, maps, ground-truth models, and wrappers.
- Learning/: training code, including CTDE-RAM implementations and related baselines.
- Evaluation/: evaluation scripts, metrics, and plotting utilities.
- runs/: experiment outputs and checkpoints.

## Main implementation

The most relevant implementation is in:

- Learning/ctde_ram/

This package contains:

- the CTDE-RAM trainer,
- role selector variants,
- reward normalization and scalarization options,
- experiment launching and evaluation utilities,
- support for both toy and real-project environments.

## Quick start

For a lightweight smoke test:

```bash
python Learning/ctde_ram/run_experiment.py --smoke --episodes 1
```

For a real-project run using the expert_nu path-planner substrate:

```bash
python Learning/ctde_ram/run_experiment.py \
  --env project \
  --project-control expert_nu \
  --path-planner-folder <folder_name> \
  --map-name malaga_port \
  --N 4 \
  --episodes 500 \
  --T-role 20 \
  --ram-mode soft_v2 \
  --soft-ram-arch attention \
  --role-state-mode pooled \
  --role-reward-norm minmax
```

## Documentation

For detailed usage, training flags, evaluation, and diagnostics, see:

- Learning/ctde_ram/README.md

## Notes

The project is designed around PyTorch and NumPy, with optional TensorBoard and Matplotlib support for training logs and plots.
