# AVE
Re-Implementation of Audio-Visual Event Localization (ECCV 2018).

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

You run the following command to start training.
```
python train.py \
    --train-path <training annotaion file path> \
    --valid-path <validation annotation file path> \
    --features-path <pre-trained audio and visual features directory path> \
    --config-path <config.yaml path>
```

If you want to evaluate the result, you run the following command.
```
python evaluate.py \
    --test-path <testing annotation file path> \
    --features-path <pre-trained audio and visual features directory path> \
    --ckpt-path <checkpoint file path to load model> \
    --yaml-path <config.yaml path> 
```

After the test, average accuracy would be displayed on your screen.
