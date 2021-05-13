import numpy as np
import pandas as pd


def load_data(directory):
    """
    Load the MNIST dataset. Reads the training or testing files and create matrices.
    :Expected return:
    data:the matrix with the training data or test data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row
    """

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

    data = data.astype(float) / 255

    return data, y
