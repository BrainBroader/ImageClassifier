from sklearn.model_selection import train_test_split
from load_data import load_cifar_10
from sga_algorithm import SgaAlgorithm
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

    # one run with standard parameters
    sga = SgaAlgorithm()
    weights = sga.train(train_data, train_target)
    pred = sga.predict(weights, x_test)
    print(np.mean(pred == np.argmax(y_test, 1)))

    # performing Grid Search

    # for function in ["tanh", "softplus", "cosine"]:
    #     for batch in [100, 200]:
    #         for hidden_layer in [100, 200, 300]:
    #             for iterations in [200, 500]:
    #                 for lr in [0.01, 1e-3]:
    #                     for lamda in [1e-2, 1e-3]:
    #                         sga = SgaAlgorithm(function, lamda, lr, hidden_layer, batch, iterations)
    #                         weights = sga.train(train_data, train_target)
    #                         pred_train = sga.predict(weights, train_data)
    #                         pred_dev = sga.predict(weights, dev_data)
    #                         print("funxtion: " + function + " mini batch: " + str(batch) + " m: " + str(hidden_layer)
    #                               + " lr: " + str(lr) + " lamda : " + str(lamda) + " iterations: " + str(iterations))
    #                         print("Accuracy for train: ")
    #                         print(np.mean(pred_train == np.argmax(train_target, 1)))
    #                         print("Accuracy for test: ")
    #                         print(np.mean(pred_dev == np.argmax(dev_target, 1)))
    #                         print("-----------------------------------")


if __name__ == '__main__':
    main()
