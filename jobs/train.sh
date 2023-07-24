#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --time=3:30:00

#SBATCH --job-name=pixel_cnn
#SBATCH --output=./out/pixel_cnn.out
#SBATCH --error=./out/pixel_cnn.err
#SBATCH --gpus=gtx_1080_ti:1

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/6.3.0 python_gpu/3.8.5 graphviz eth_proxy
source venv/bin/activate

python src/main.py
