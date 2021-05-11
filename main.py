from load_data import load_data
from sga_algorithm import train
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    costs, w1, w2, b1, b2 = train(X_train, y_train)
    print(costs)
