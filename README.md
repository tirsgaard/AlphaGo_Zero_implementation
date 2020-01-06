# AlphaGo Zero implementation
Implementation of AlphaGo Zero for a course.
The ResNet Block code is from the great blogpost https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
The elo system is based on the fantastic implementation here https://github.com/HankSheehan/EloPy .

Everything else is written by us.

## Packages used
Pytorch is needed.
To visualize the learning we use the tensorboard, so this has to be installed if running "main.py". This is not needed if you only want to run "play_against_newest_model.ipynb".

## Running the code
The file main_file.py is the master function calling the self-play, training, and evaluation functions.
This takes ages to run, so we also have some tester functions, to run or test the different implementations.

We recommend playing against the best model in the notebook "play_against_newest_model.ipynb".
To do this create a directory called "saved_models" in the same directory as this implementation.
Then download the model paremters from https://drive.google.com/open?id=12iWse0HQL9CItQPwc_FLNRIt57VGi5UU or use the included "model11.model",
Finally move the model file to new directory.

Some hyperparameters has to be changed depending on the computer running it such as number of threads.
It takes ~24h to complete 2,000 self-play games using a AMD threadripper 1950x processer and a Titan V GPU.
