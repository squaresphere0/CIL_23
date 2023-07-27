#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --time=3:30:00

#SBATCH --mem-per-cpu=32g

#SBATCH --job-name=inference_trans
#SBATCH --output=./out/inference_trans.out
#SBATCH --error=./out/inference_trans.err

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/8.2.0 python_gpu/3.11.2 graphviz eth_proxy
source venv/bin/activate

python -u src/transformer_create_mask.py
