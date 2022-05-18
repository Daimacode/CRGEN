# CRGEN

# Environments requirements: 
pytorch-1.4.0,torchvision-0.5.0,numpy

# Dataset
Before running the code, you need to prepare the dataset MNIST-rot and put it in path '../data/mnist_rot/train.pt' and '../data/mnist_rot/test.pt', or you can set your own path by changing the path in the code. 

# Run
python structModel_test.py -gpu 0 --name=test
