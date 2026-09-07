# Author-run reference evidence

These are the supplied reference implementation results, not the learner's work.
Environment: Python 3.12, PyTorch 2.14.0+cpu, one CPU thread for training.
31 Gaussians; synthetic 128x128 targets downsampled to 64x64; 500 updates.
Multi-view: 8 training views. One-view: view 0 only. Four shared held-out views.
Same helpful initialization, optimizer settings, and update count in both.

- Covariance, projection, weights, blending, finite difference, and bounded/dense
  cutoff agreement checks passed.
- Both 500-update training runs completed. See metrics.json and config.json.
- Basic-only optimization smoke test passed.
- 512x512 model-load/refinement smoke test passed (2 updates; not a quality benchmark).
- Checkpoint evaluation, PNG/GIF export, and ablation commands ran successfully.
- 256x256 reference benchmark: about 0.180 seconds/update on this environment's CPU,
  20 measured updates after 5 warm-up updates. Other work was running concurrently;
  treat this as a rough local measurement, not a hardware comparison.
- No RunPod deployment or NVIDIA GPU execution was performed by the guide author.
- The learner core.py intentionally remains unfinished.

The default full guide path generates fixed targets at 512x512. Low-resolution
validation here is only evidence that the small training loop is functional.
