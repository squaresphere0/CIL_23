#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

#SBATCH --job-name=pixel_cnn
#SBATCH --output=./out/pixel_cnn.out
#SBATCH --error=./out/pixel_cnn.err

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
source venv/bin/activate

python src/main.py
