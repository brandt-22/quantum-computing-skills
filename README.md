# Quantum Computing / AMD ROCm Devin Skills

Reusable [Devin CLI skills](https://docs.devin.ai) distilled from a series of
QSVT (Quantum Singular Value Transformation) experiments on AMD Instinct
MI355X GPUs using PennyLane, including a real CFD matrix extracted from
OpenFOAM. See the companion experiment/code repo:
[qsvt-mi355x-openfoam-experiments](https://github.com/brandt-22/qsvt-mi355x-openfoam-experiments).

## Skills

- **`pennylane-amdgpu-qsvt`** ??? Build and benchmark QSVT circuits with
  PennyLane on AMD Instinct GPUs (`lightning.amdgpu`), including how to build
  a genuinely sparsity-aware LCU block encoding (PennyLane's generic
  `qml.BlockEncode` does *not* exploit matrix sparsity and will not scale).
- **`rocm-multi-gpu-pinning`** ??? Correctly pin a containerized workload to a
  specific physical AMD GPU on a multi-GPU ROCm host. Covers a real,
  non-obvious gotcha: setting both `HIP_VISIBLE_DEVICES` and
  `ROCR_VISIBLE_DEVICES` together causes a GPU-index collision; only
  `ROCR_VISIBLE_DEVICES` alone gives a correct mapping.
- **`openfoam-matrix-extraction`** ??? Extract the real sparse linear-system
  (LDU) matrix from an OpenFOAM CFD solve, for use in external
  linear-algebra/quantum-computing tools. Includes a working solver-patch
  approach (simpler than the third-party `openfoam-matrix-export` library,
  which has an awkward nested git-submodule dependency chain).

## Installation

Copy the `.devin/skills/<name>/` directories into your project's `.devin/skills/`
directory (project-scoped) or `~/.config/devin/skills/` / `%APPDATA%\devin\skills\`
(global), matching the structure in this repo.

