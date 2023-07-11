# PixelCNN for Road Segmentation

## File layout for Petr

The python scripts expect the following directory layout.

```
.
├─ src/
|  └─ sourcefiles as in git repo
├─ Datasets/
|  └─ ethz─cil─road─segmentation─2023/
|     └─ Data folder as is found on __Augusts Drive__
├─ model/
└─ train.sh
```

All python scripts expect to be executed from the root directory for correctly applying their paths.

The model folder can be empty but **must** exist because it's where the script will save the trained model.

I've used the train.sh that you supplied and simply changed `main.py` to `src/main.py`, which worked fine for me.
