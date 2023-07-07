# CIL_23

## Setting up on Euler cluster

This **should be done once** before running any scripts, otherwise just remove the "venv" folder with `rm -rf venv` and start over
```
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
python -m venv venv --system-site-packages
export SETUPTOOLS_USE_DISTUTILS=stdlib

```

#### To launch with SLURM
- from the root directory: `cd jobs`
- `sbatch main_patch_cnn.sh`
- check status with: `for i in {1..20}; do squeue; sleep 2; done` — (use CTRL+C to stop it)
- (optional) cancel with: `scancel <jobid>`
