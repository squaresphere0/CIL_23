# PixelCNN for Road Segmentation

## Using this Repo

### Running the Code

The following list explains roughly the use of each source code file and whoch of them are needed to re runn the experiments from the projects.

|Filename | description |
|---------|-------------|
|u_net.py | Contains the CIL implementation of a U-NET.|
|patch_cnn.py| Contains the CIL implementation of patch CNN.|
|main_baselines.py | Runnable script to train either of the baselines. Expects either `--baseline=patch_cnn` or `--baseline=unet` as a argument.|
|training_loop.py | Contains the trining loop used to train the baselines. |
|utils.py | Some utility code for the baselines. |
|mask_to_submission.py | As provided for the project. |
|submission_to_mask.py | As provided for the project. |
|pixel_cnn.py | Contains our implementation of a conditional PixelCNN including the training loop.|
|dataloader.py | Contains a lazydataloader implementation used for the training of our PixelCNN. |
|main.py | Used to setup training sessions of our PixelCNN. All relevant hyperparameters are exposed either at the initialization of the PixelCNN or at the method call to train it. Saves any trained model into the model directory.|
|pixelcnn_create_mask.py | Contains the code to load a trained PixelCNN model and generate predictions for the ethz test set. These are saved as images into a directory by the same name as the model inside of the test directory. |
|playground.py | Contains some utility functions used during the project to asses PixelCNN models.|

All python scripts expect to called from the root directory of the repo. Otherwise the relative paths used will not checkout to the correct locations.

#### Directory Layout

The python scripts expect the following directory layout.

```
.
├─ src/
├─ Datasets/
|  └─ ethz─cil─road─segmentation─2023/
|  └─ DeepGlobe/
├─ model/
└─ jobs/
```

All python scripts expect to be executed from the root directory for correctly applying their paths.

The model folder can be empty but **must** exist because it's where the script will save the trained model.

### The Datasets

We have removed the subdirectories of the datasets holding the images for the repo. Our only addition to the datasets are the metadata.csv files used for loading the data by our code.

This means the original Dataset Folders can be used as long as the csv files are copied over.

The Deepglobe dataset can be sourced [here](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset).

### The Bash scripts

In the jobs directory bash scripts exist to trin some of the model on euler.

To pred the evironment this **should be done once** before running any scripts, in the root directory of the repo.
Otherwise just remove the "venv" folder with `rm -rf venv` and start over.
```
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
python -m venv venv --system-site-packages
export SETUPTOOLS_USE_DISTUTILS=stdlib
pip install comet_ml
pip install torchview
pip install cairosvg

```

The scripts expect to be queued from within the jobs directory.


