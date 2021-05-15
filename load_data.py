import numpy as np
import pandas as pd
import pickle


def load_mnist(directory):
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

    data = data.astype(np.float64) / 255

    return data, y


def load_cifar_10(directory):

    x_train = []
    y_train = []

    for i in range(1, 6):

        tmp = open_cifar_10(directory + "/data_batch_" + str(i))
        x_train.extend(tmp[b'data'])
        y_train.extend(tmp[b'labels'])

    tmp = open_cifar_10(directory + "/test_batch")
    x_test = np.copy(tmp[b'data'])
    y_test = tmp[b'labels']

    train_data = np.array(x_train)
    test_data = np.array(x_test)
    print(train_data.shape)
    print(test_data.shape)

    class_names = open_cifar_10(directory + "/batches.meta")

    train_target = np.zeros([len(y_train), 10])

    for i in range(train_target.shape[0]):
        train_target[i, y_train[i]] = 1

    test_target = np.zeros([len(y_test), 10])
    for i in range(test_target.shape[0]):
        train_target[i, y_test[i]] = 1

    return train_data, train_target, test_data, test_target, class_names


def open_cifar_10(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')

        return dic
