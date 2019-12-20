# weakly-supervised-segmentation
This is a repository for our final project of CIE 6004 `weakly supervised medical image segmentation`.

## Prerequisites
* python 3
* PyTorch 1.1.0

## Data Preparation
* Download data from `https://kits19.grand-challenge.org/`
* Create a folder: `mkdir data`
* Put the dataset to data, i.e. `data/kidney_original`

## Getting Started
### Training without boundary constraint loss
For kidney dataset, run:

`
python train.py -c config/config_kidney.json
`

### Testing without boundary constraint loss
For kidney dataset, run:

`
python infer_cls.py -c config/config_kidney.json -r <model path>
`

### Training with boundary constraint loss
For kidney dataset, run:

`
python train_bc.py -c config/config_kidney.json
`

### Testing with boundary constraint loss
For kidney dataset, run:

`
python infer_bc_cls.py -c config/config_kidney.json -r <model path>`

## Project Structure

`base/`: abstract base classes

`data_loader/`: anything about data loading goes here

`data/`: default directory for storing input data

`model/`: models, losses, and metrics

`config/`: default directory for configuration file

`saved/`: default directory for storing output data

`trainer/`: training epoch

`logger/`: module for tensorboard visualization and logging

`utils/`: small utility functions

`refine/`: code for post-precessing written in matlab