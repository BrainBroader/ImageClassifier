import numpy as np


def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)

    return p / np.sum(p, axis=ax, keepdims=True)


def cost_grad_softmax(W, X, t, lamda):
    # X: NxD
    # W: KxD
    # t: NxD

    E = 0
    N, D = X.shape
    K = t.shape[1]

    y = softmax(np.dot(X, W.T))

    for n in range(N):
        for k in range(K):
            E += t[n][k] * np.log(y[n][k])
    E -= lamda * np.sum(np.square(W)) / 2

    grad_ew = np.dot((t - y).T, X) - lamda * W

    return E, grad_ew
