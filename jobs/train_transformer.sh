#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus=rtx_3090:1            # --gpus=rtx_3090:1 or --gpus=a100-pcie-40gb:1

#SBATCH --job-name=just_a_tr
#SBATCH --output=./out/just_a_tr.out
#SBATCH --error=./out/just_a_tr.err

cd ..

rm -rf preds/*

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/8.2.0 python_gpu/3.9.9 graphviz eth_proxy
source venv/bin/activate

python -u src/just_a_transformer.py
