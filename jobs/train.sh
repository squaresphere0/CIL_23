#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --time=3:30:00
#SBATCH --mem-per-cpu=32g

#SBATCH --job-name=pixel_cnn
#SBATCH --output=./out/pixel_cnn.out
#SBATCH --error=./out/pixel_cnn.err
#SBATCH --gpus=gtx_1080_ti:1

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/8.2.0 python_gpu/3.11.2 graphviz eth_proxy
source venv/bin/activate

python -u src/main.py
