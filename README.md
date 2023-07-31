# PixelCNN for Road Segmentation

## Initial environment Setup on Euler

Use the following commands in the project **root folder** to setup the environment. This **should be done once** before running any scripts.
Otherwise remove the "venv" folder with `rm -rf venv` and start over.
```
export SETUPTOOLS_USE_DISTUTILS=stdlib
module load gcc/8.2.0 python_gpu/3.11.2 graphviz eth_proxy
python -m venv venv --system-site-packages
source venv/bin/activate


pip install timm
pip install huggingface_hub -U

pip install comet_ml
pip install torchview
pip install graphviz
pip install cairosvg

pip install segmentation_models_pytorch

```

#### The Bash scripts

The bash scripts in the "jobs" are used to train models on euler.
The scripts are expected to be run from within the "jobs" directory.

I. e. `pwd` outputs `.../jobs` then run:
```
sbatch train_transformer.sh
```
For the transformer model it is also possible to run inference with the "inference_create_masks_parallel.sh" script (but the name of the model from "model/" folder should be changed accordingly in the script).


## Using this Repo

### Running the Code

The following list explains roughly the use of each source code file and whoch of them are needed to re runn the experiments from the projects.

|Filename | description |
|---------|-------------|
|pixel_cnn.py | Contains our implementation of a conditional PixelCNN including the training loop.|
|main.py | Used to setup training sessions of our PixelCNN. All relevant hyperparameters are exposed either at the initialization of the PixelCNN or at the method call to train it. Saves any trained model into the model directory.|
|pixelcnn_create_mask.py | Contains the code to load a trained PixelCNN model and generate predictions for the ethz test set. These are saved as images into a directory by the same name as the model inside of the test directory. |
|playground.py | Contains some utility functions used during the project to asses PixelCNN models.|
|---|---|
|transformer.py| Used for training swin2 with skip connections. |
|generate_more_data_from_deepglobe.py| This code is used to generate data from the DeepGlobe dataset. It crops images of 1024x1024 into four images of 400x400. |
|src/transformer_create_mask.py| This code is used to perform inference with the transformer model. |
|---|---|
|u_net.py | Contains the CIL implementation of a U-NET.|
|patch_cnn.py| Contains the CIL implementation of patch CNN.|
|main_baselines.py | Runnable script to train either of the baselines. Expects either `--baseline=patch_cnn` or `--baseline=unet` as a argument.|
|training_loop.py | Contains the trining loop used to train the baselines. |
|utils.py | Some utility code for the baselines. |
|mask_to_submission.py | As provided for the project. |
|submission_to_mask.py | As provided for the project. |
|dataloader.py | Contains a lazydataloader implementation used for the training of our PixelCNN. |

All python scripts expect to called from the root directory of the repo. Otherwise the relative paths used will not checkout to the correct locations.

#### Directory Layout

The python scripts expect the following directory layout.

```
.
├─ src/
├─ Datasets/
|  ├─ ethz─cil─road─segmentation─2023/
|  ├─ DeepGlobe/
|  └─ data/
|     ├─ training/
|     |  ├─ images/
|     |  └─ groundtruth/
|     └─ validation/
|        ├─ images/
|        └─ groundtruth/
├─ model/
└─ jobs/
```


All python scripts expect to be executed from the root directory for correctly applying their paths.

The model folder can be empty but **must** exist because it's where the script will save the trained model.

### The Datasets

We have removed the subdirectories of the datasets holding the images for the repo. Our only addition to the datasets are the metadata.csv files used for loading the data by our code.

(!) The `src/main_transformer.py` expects to have a Datasets/data/ folder with the provided above structure (for the most of our experiments with transformer, we used the first 11 pictures for validation, and the others for training from the original Kaggle dataset).

We also used the DeepGlobe dataset with the `src/generate_more_data_from_deepglobe.py` script, the list of files is provided in `Datasets/data/my_dataset_from_deepglobe_plus_ethz.txt`.

This means the original Dataset Folders can be used as long as the csv files are copied over.

The Deepglobe dataset can be sourced [here](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset).

