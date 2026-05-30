# Adding PinsFaceRecognition Dataset

PinsFaceRecognition is one of the 5 datasets used in the paper. Unlike CIFAR-10/20/100 (which are downloaded automatically by PyTorch), PinsFaceRecognition must be downloaded manually from Kaggle.

## Steps

1. Sign in to [Kaggle](https://www.kaggle.com/)
2. Download the [PinsFaceRecognition dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) by clicking `Download > Download dataset as zip (390 MB)`
3. Extract the zip file and place the `105_classes_pins_dataset` directory under `supreme/datasets/data/` in the codebase

**Note:** This dataset is already supported by the framework's configuration in [supreme/utils/project_config.py](../supreme/utils/project_config.py), so no code changes are needed once the data is placed in the correct location.
