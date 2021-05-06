from load_data import load_data
from sga_algorithm import train
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    train(X_train, y_train, 20, 10, 200, 0.01)
