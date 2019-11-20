# weakly-supervised-segmentation
This is a repository for our final project of CIE 6004.

## Prerequisites
* python 3
* PyTorch 1.1.0

## Data Preparation
* Create a folder: `mkdir data`
* Put the dataset to data, i.e. `data/abdomen` and `data/kidney`

## Getting Started
### Training
For abdomen dataset, run:

`
python train.py -c config_abdomen.json
`

For kidney dataset, run:

`
python train.py -c config_kidney.json
`

### Testing

For abdomen dataset, run:

`
python test.py -c config_abdomen.json -r <model path>
`

For kidney dataset, run:

`
python test.py -c config_kidney.json -r <model path>
`

## Project Structure

`base/`: abstract base classes

`data_loader/`: anything about data loading goes here

`data/`: default directory for storing input data

`model/`: models, losses, and metrics

`saved/`: default directory for storing output data

`trainer/`: training epoch

`logger/`: module for tensorboard visualization and logging

`utils/`: small utility functions