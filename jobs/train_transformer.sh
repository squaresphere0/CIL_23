#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=32g
#SBATCH --job-name=tr_w_unet
#SBATCH --output=./out/tr_w_unet.out
#SBATCH --error=./out/tr_w_unet.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user="mrpetrkol@gmail.com"

cd ..

export SETUPTOOLS_USE_DISTUTILS=stdlib

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
source venv/bin/activate

python src/transformer_with_unet.py
