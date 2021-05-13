import sys

from sklearn.model_selection import train_test_split

from load_data import load_data
from test import train
from sga_algorithm import SgaAlgorithm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train_path ='ml/train%d.txt' # sys.argv[1] + '\\train%d.txt'
    test_path ='ml/test%d.txt' # sys.argv[1] + '\\test%d.txt'

    # load training data
    print(f'[INFO] - Loading training data from {train_path}')
    X_train, y_train = load_data(train_path)
    print(f'[INFO] - Total train data: {len(X_train)}')

    print(f'[INFO] - Loading testing data from {test_path}')
    X_test, y_test = load_data(test_path)

    print(f'[INFO] - Total test data: {len(X_test)}')

    # 10% of training data will go to developer data set
    print(f'[INFO] - Splitting training data into training data and developer data (keeping 10% for training data)')
    res = train_test_split(X_train, y_train, test_size=0.1)
    train_data = res[0]
    train_target = res[2]
    print(f'[INFO] - Total training data after split {len(train_data)}')
    dev_data = res[1]
    dev_target = res[3]
    print(f'[INFO] - Total developer data {len(dev_data)}')

    sga = SgaAlgorithm()
    costs, w1, w2, b1, b2 = sga.train(train_data, train_target)
    pred = sga.predict(w1, w2, b1, b2, dev_data)

    print(np.mean(pred == np.argmax(dev_target, 1)))




