import numpy as np


def train(x, t, m=500, mini_batch=200, iterations=1000, lr=0.01, tol=0.00001):

    # create mini batch and get vector's dimensions
    x, t = create_batch(x, t, mini_batch)
    n, d = x.shape
    k = t.shape[1]

    # random initialization of weights and bias
    w1 = np.random.randn(d, m)
    w2 = np.random.randn(m, k)
    b1 = np.random.randn(m)
    b2 = np.random.randn(k)

    costs = []
    e_wold = -np.inf

    for i in range(iterations):
        x, t = create_batch(x, t, mini_batch)

        # forward step
        z = h(x.dot(w1) + b1)
        y = softmax(z.dot(w2) + b2)

        # calculate cost function
        ew = 0
        for n in range(n):
            for k in range(k):
                ew += t[n][k] * np.log(y[n][k])
        ew -= lr * np.sum(np.square(w2)) / 2

        if np.abs(ew - e_wold) >= tol:

            # backpropagation
            delta2 = t - y
            delta1 = delta2.dot(w2.T) * z * (1 - z)

            w2 += lr * (z.T.dot(delta2) - lr * w2)
            b2 += lr * delta2.sum(axis=0)

            w1 += lr * (x.T.dot(delta1) - lr * w1)
            b1 += lr * delta1.sum(axis=0)

            # save loss function values across training iterations
            if i % 50 == 0:
                print('Loss function value: ', ew)
                costs.append(ew)
        else:
            break

    return costs, w1, w2, b1, b2


def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)

    return p / np.sum(p, axis=ax, keepdims=True)


def h(z):
    return 1 / (1 + np.exp(-z))


def h_derivative(z):
    return np.exp(z) / (1 + np.exp(z))


def create_batch(x, y, mini_batch):

    random_indices = np.random.choice(x.shape[0], size=mini_batch, replace=False)
    x_mini = x[random_indices, :]
    y_mini = y[random_indices, :]

    return x_mini, y_mini
