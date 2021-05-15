from sklearn.model_selection import train_test_split

from load_data import load_cifar_10

from sga_algorithm import SgaAlgorithm
import matplotlib.pyplot as plt
import numpy as np


def main():

    # Load data
    path = "cifar-10-batches-py"

    print(f'[INFO] - Loading training data from {path}')
    x_train, y_train, x_test, y_test, class_names = load_cifar_10(path)

    # 10% of training data will go to developer data set
    print(f'[INFO] - Splitting training data into training data and developer data (keeping 10% for training data)')
    res = train_test_split(x_train, y_train, test_size=0.1)
    train_data = res[0]
    train_target = res[2]
    print(f'[INFO] - Total training data after split {len(train_data)}')
    dev_data = res[1]
    dev_target = res[3]
    print(f'[INFO] - Total developer data {len(dev_data)}')

    sga = SgaAlgorithm()
    weights = sga.train(train_data, train_target)
    pred = sga.predict(weights, train_data)
    print(np.mean(pred == np.argmax(train_target, 1)))


if __name__ == '__main__':
    main()
