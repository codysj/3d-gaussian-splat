# Copy-friendly command sheet

Run all commands from the directory containing run.py.

## Windows PowerShell setup
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py check --stage covariance
```
For other local commands below, replace `python` with `.\.venv\Scripts\python.exe`
and omit `--device cuda` (or use `--device cpu`). No activation is needed.

## Checks, in learning order
```bash
python run.py check --stage covariance
python run.py check --stage projection
python run.py check --stage weights
python run.py check --stage blend
python run.py check --stage gradient
python run.py check
python run.py make --size 64 --out outputs/smoke
```

## RunPod: keep the template's installed PyTorch
```bash
cd /workspace/gaussian_workbook
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python run.py check --device cuda
python -m pip freeze > environment-runpod.txt
```

## Targets, benchmarks, and training
```bash
python run.py make --device cuda --size 512 --out outputs/data
python run.py bench --device cuda --size 128 --steps 100
python run.py bench --device cuda --size 256 --steps 100
python run.py bench --device cuda --size 512 --steps 100
python run.py train --device cuda --size 128 --steps 200 --learn basic --out outputs/basic
python run.py train --device cuda --size 256 --steps 1200 --learn all --out outputs/main
python run.py train --device cuda --size 512 --steps 400 --resume outputs/main/checkpoint.pt --lr-factor 0.25 --out outputs/refine
```
Refinement restarts Adam using the trained parameters and lower learning rates.

## Controlled experiments and common-resolution evaluation
```bash
python run.py train --device cuda --size 256 --steps 1200 --views one --out outputs/one
python run.py ablate --device cuda --size 256 --checkpoint outputs/main/checkpoint.pt --out outputs/ablation
python run.py eval --device cuda --size 512 --checkpoint outputs/main/checkpoint.pt --out outputs/eval_main
python run.py eval --device cuda --size 512 --checkpoint outputs/refine/checkpoint.pt --out outputs/eval_refine
```

## Export (substitute main if there is no refinement run)
```bash
python run.py export --device cuda --size 512 --frames 36 --checkpoint outputs/refine/checkpoint.pt --out outputs/demo
python -m zipfile -c results.zip outputs core.py support.py run.py NOTES.md environment-runpod.txt
```
Download results.zip via Jupyter, open it locally, then stop/terminate the Pod as
appropriate. A network volume has a separate lifecycle and storage charges.

## Author's reference validation reproduction (not the learner path)
```bash
python run.py check --reference --threads 1
python run.py make --reference --size 128 --out outputs/validation_data --threads 1
python run.py train --reference --size 64 --steps 500 --threads 1 --data outputs/validation_data/dataset.npz --out outputs/validation_multi
python run.py train --reference --size 64 --steps 500 --threads 1 --views one --data outputs/validation_data/dataset.npz --out outputs/validation_one
```

## Gnome scene on RunPod (GPU; dense memory is about 1 MB per Gaussian at 256 px)
Template: RunPod PyTorch (CUDA). Keep its torch; do NOT pip install torch from requirements.txt.
```bash
cd /workspace && git clone https://github.com/codysj/3d-gaussian-splat gaussian_workbook && cd gaussian_workbook
pip install numpy matplotlib pillow plyfile trimesh
nvidia-smi && python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python run.py check --device cuda
python scene_from_mesh.py assets/garden_gnome/garden_gnome_1k.gltf --count 10000 --out assets/gnome_10000.npz
python run.py make --device cuda --size 256 --scene assets/gnome_10000.npz --out outputs/gnome_data
python run.py train --device cuda --size 256 --steps 2000 --data outputs/gnome_data/dataset.npz --out outputs/gnome_main
python run.py eval --device cuda --size 256 --checkpoint outputs/gnome_main/checkpoint.pt --data outputs/gnome_data/dataset.npz --out outputs/gnome_eval
python run.py ablate --device cuda --size 256 --checkpoint outputs/gnome_main/checkpoint.pt --out outputs/gnome_ablation
python run.py export --device cuda --size 384 --frames 36 --checkpoint outputs/gnome_main/checkpoint.pt --out outputs/gnome_demo
python -m pip freeze > environment-runpod.txt
python -m zipfile -c gnome_results.zip outputs/gnome_main outputs/gnome_data/dataset.npz outputs/gnome_eval outputs/gnome_ablation outputs/gnome_demo/orbit.gif environment-runpod.txt
```
Download gnome_results.zip through the Jupyter file browser, then stop the Pod.
`bench` uses the 31-Gaussian robot and says nothing about gnome cost; read the
per-100-update timestamps from `train` instead.
