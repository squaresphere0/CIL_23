#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

#SBATCH --job-name=just_a_tr
#SBATCH --output=./out/just_a_tr.out
#SBATCH --error=./out/just_a_tr.err

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
source venv/bin/activate

python -u src/just_a_transformer.py
