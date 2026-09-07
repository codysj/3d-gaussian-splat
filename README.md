# Gaussian rendering workbook starter

Read Gaussian_Rendering_Workbook.pdf in order. This project is a teaching scaffold,
not a finished project the learner should claim to have independently authored.

## First action

Install the CPU PyTorch build locally and the packages in requirements.txt (see
COMMANDS.md). Fill TODO 1 in core.py, then run `python run.py check --stage covariance`.
The five core functions intentionally raise NotImplementedError until implemented.
Later commands use your functions by default.

## Supplied versus learner work

- **You implement:** covariance, projection, pixel influence, compositing, training update.
- **Supplied:** model/scene helpers, camera and quaternion helpers, renderer wiring,
  bounded forward reference, CLI, optimizer groups, experiments, I/O, checks.
- **Answer key:** reference/core.py. `--reference` explicitly chooses those answers.
- **Additional exercise:** reconstruct bounded_forward in support.py. Dense training
  remains separate because the supplied bounded path is forward-only.

Use assistance honestly. Understanding and editing supplied code is different from
independently implementing it. Record what you reused in NOTES.md.

## Scope

The default robot uses 31 Gaussians and fixed RGB. Targets are produced by the same
renderer. Initialization is a perturbation of target parameters with a known count.
Training compares images only, but the initialization supplies substantial geometry.
There is no COLMAP, unknown-camera recovery, random-initialized full reconstruction,
densification, pruning, spherical-harmonic color, or custom CUDA kernel here.

The independent numeric checks reduce the shared-renderer blind spot. The held-out
test measures new-camera agreement for one scene, not new-scene generalization.

## Commands

See COMMANDS.md. All output paths are relative to the current project directory.
`--resume` loads parameters only and restarts Adam deliberately. The checkpoint also
contains optimizer state for inspection, but the refinement CLI does not restore it.
Load only trusted checkpoints. Dense runtime/memory scales with Gaussians times pixels.

## Validation

validation/ contains author-run reference evidence using PyTorch 2.14.0+cpu.
These results are not learner results, and no RunPod GPU execution was performed.
The PDF includes the measured conditions. Numeric checks, small training, export,
ablation, and evaluation paths were exercised. Your core.py remains unfinished.

## Useful primary sources

- https://arxiv.org/html/2308.04079v1 (full original method, sections 4-6)
- https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- https://docs.runpod.io/get-started
- https://docs.runpod.io/pods/connect-to-a-pod
- https://docs.runpod.io/pods/storage/types

