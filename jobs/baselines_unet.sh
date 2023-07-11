#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=32g
#SBATCH --job-name=unet_baseline
#SBATCH --output=./out/unet_baseline.out
#SBATCH --error=./out/unet_baseline.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
source venv/bin/activate

python src/main_baselines.py --baseline="unet"
