import numpy as np


def train(x, t, m, k, mini_batch, lr):

    # create mini batch and get vector's X dimensions
    x, t = create_batch(x, t, mini_batch)
    n, d = x.shape
    # random initialization of weights
    w1 = np.random.randn(m, d)
    w2 = np.random.randn(k, m)
    print(x.shape)
    # forward step
    z = h(x.dot(w1.transpose()))
    y = softmax(z.dot(w2.transpose()))

    # calculate cost function
    ew = 0
    for n in range(n):
        for k in range(k):
            ew += t[n][k] * np.log(y[n][k])
    ew -= lr * np.sum(np.square(w2)) / 2

    # backpropagation
    temp = (t-y).transpose()
    w2 += lr*(temp.dot(z) - lr*w2)

    # w1 += lr*(temp.dot(w2.transpose()).dot(x).dot(h_derivative(x.dot(w1.transpose()))))


def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)

    return p / np.sum(p, axis=ax, keepdims=True)


def h(z):
    return np.log(1 + np.exp(z))


def h_derivative(z):
    return np.exp(z) / (1 + np.exp(z))


def create_batch(x, y, mini_batch):

    random_indices = np.random.choice(x.shape[0], size=mini_batch, replace=False)
    x_mini = x[random_indices, :]
    y_mini = y[random_indices, :]

    return x_mini, y_mini
