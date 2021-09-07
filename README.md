# ImageClassifier

Classifying images using a MLP implemented from scratch.

**University:** Athens University of Economics and Business  
**Department:** Informatics  
**Subject:** Machine Learning

**Writer:**  Andreas Gouletas (@BrainBroader)

## Table of Contents
* [Description](#description)
* [Datasets](#datasets)
* [Prerequisites](#prerequisities)
* [Execution Instructions](#execution-instructions)

### Description 
We implement a simple Multilayer Perceptron(MLP) from scratch for classification and train it using the gradient ascent algorithm . Our MLP has a hidden layer that uses tanh or softplus or cosine as activation function. At the output layer our MLP uses the softplus activation function.

### Datasets
1. The MNIST dataset: 
Is a dataset that contains 28x28 grayscale images of numbers from 0-9. So we classify the images in 10 categories. The data is stored in txt files.

Download: http://yann.lecun.com/exdb/mnist/

2. The CIFAR-10 dataset:
Is a dataset that contains 32x32 colored images of different objects or animals. There are 10 different categories and we read the data from txt files.

Download: https://www.cs.toronto.edu/~kriz/cifar.html

### Technologies

The technologies used that are worth mentioning, are:

   * Python
   * Numpy


### Prerequisities

Before you execute the given program, you need to:

    1.download and unzip the dataset you want,
    2.check if you have installed the libraries mention in Section "Technologies".

If you haven't previously installed the libraries mentioned above, you can use the provided requirements.txt file, by running the following command:

cd path-to-project
pip install -r requirements.txt
 
### Execution Instructions
To execute the program the following command is used:
```
python xxx_main.py arg1
```
where 
* xxx_main is one of the two main running scripts (mnist_main.py, cifar_10_main.py),
* arg2 is the path to the MNIST or CIFAR-10 dataset.


Running for MNIST, 
```
python mnist_main.py \\MNIST
```

Running for CIFAR-10, 
```
python cifar_10_main.py \\cifar-10
