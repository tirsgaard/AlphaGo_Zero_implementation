# AlphaGo Zero implementation
Implementation of AlphaGo Zero for a course.
The ResNet Block code is from the great blogpost https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
The elo system is based on the fantastic implementation here https://github.com/HankSheehan/EloPy .

Everything else is written by us.

## Packages used
To visualize the learning we use the tensorboard, so this has to be installed.

## Running the code
The file main_file.py is the master function calling the self-play, training, and evaluation functions.
This takes ages to run, so we also have some tester functions, to run or test the different implementations.

You can play against the best model in the notebook "play_against_newest_model.ipynb".

Some hyperparameters has to be changed depending on the computer running it such as number of threads.
It takes ~24h to complete 2,000 self-play games using a AMD threadripper 1950x processer and a Titan V GPU.
