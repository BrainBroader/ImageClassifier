import numpy as np
import pandas as pd


def load_data():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    train_data, y_train = load('ml/train%d.txt')
    test_data, y_test = load('ml/test%d.txt')

    train_data = train_data.astype(float) / 255
    test_data = test_data.astype(float) / 255

    #train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    #test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    return train_data, test_data, y_train, y_test


def load(directory):

    df = None

    y = []

    for i in range(10):
        tmp = pd.read_csv(directory % i, header=None, sep=" ")
        # build labels - one hot vector

        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    data = df.to_numpy()
    y = np.array(y)

    return data, y
