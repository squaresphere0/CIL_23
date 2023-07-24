# PixelCNN for Road Segmentation

## File layout for Petr

The python scripts expect the following directory layout.

```
.
├─ src/
|  └─ sourcefiles as in git repo
├─ Datasets/
|  └─ ethz─cil─road─segmentation─2023/
|     └─ Data folder as is found on __Augusts Drive__
├─ model/
└─ train.sh
```

All python scripts expect to be executed from the root directory for correctly applying their paths.

The model folder can be empty but **must** exist because it's where the script will save the trained model.

I've used the train.sh that you supplied and simply changed `main.py` to `src/main.py`, which worked fine for me.

## Setting up on Euler cluster

This **should be done once** before running any scripts, otherwise just remove the "venv" folder with `rm -rf venv` and start over
```
export SETUPTOOLS_USE_DISTUTILS=stdlib
module load gcc/8.2.0 python_gpu/3.9.9 graphviz eth_proxy
python -m venv venv --system-site-packages
source venv/bin/activate


pip install timm
pip install huggingface_hub -U

pip install comet_ml
pip install torchview
pip install graphviz
pip install cairosvg

```

#### To launch with SLURM
- from the root directory: `cd jobs`
- `sbatch main_patch_cnn.sh`
- check status with: `for i in {1..20}; do squeue; sleep 2; done` — (use CTRL+C to stop it)
- (optional) cancel with: `scancel <jobid>`
