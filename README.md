# AVE
Re-Implementation of Audio-Visual Event Localization (ECCV 2018)

## Requirements

### Python modules
- torch
- torchvision
- resampy
- soundfile
- PyAV

### OS packages

- ffmpeg
- libsndfile1

## Run

### Audio Visual Event Localization

You run the following command to start training.
```
python avel_main.py \
    --ave-root <dataset root directory path> \
    --train-annot <training annotaion file path> \
    --valid-annot <validation annotation file path> \
    --feature-path <pre-trained audio and visual features directory path> \
    --yaml-path <config.yaml path>
```

If you want to extract audio and visual features before training, you can use ```--extract-feature``` flag with above command.
```
python avel_main.py \
    --ave-root <dataset root directory path> \
    --train-annot <training annotaion file path> \
    --valid-annot <validation annotation file path> \
    --feature-path <pre-trained audio and visual features directory path> \
    --yaml-path <config.yaml path> \
    --extract-feature
```

After the training, result figure image which contains loss and accuarcy over training would be saved at this project root.
Also the checkpoint of training model is saved under the ckpt directory.

If you want to run test, you run the following command.
```
python avel_test.py \
    --ave-root <dataset root directory path> \
    --test-annot <testing annotation file path> \
    --feature-path <pre-trained audio and visual features directory path> \
    --yaml-path <config.yaml path> \
    --ckpt-path <checkpoint file path to load model>
```

After the test, average accuracy would be displayed on your screen.
