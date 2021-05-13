import numpy as np


class SgaAlgorithm:

    def __init__(self, m=700, mini_batch=400, iterations=1000, lr=0.05, tol=1e-6):
        self.m = m
        self.mini_batch = mini_batch
        self.iterations = iterations
        self.lr = lr
        self.tol = tol

    def train(self, x, t):

        d = x.shape[1]
        k = t.shape[1]

        # random initialization of weights and bias
        w1 = np.random.randn(d, self.m)
        w2 = np.random.randn(self.m, k)
        b1 = np.random.randn(self.m)
        b2 = np.random.randn(k)

        costs = []
        e_wold = -np.inf

        for i in range(self.iterations):
            x, t = self.create_batch(x, t)

            # forward step
            z = self.h(x.dot(w1) + b1)
            y = self.softmax(z.dot(w2) + b2)

            # calculate cost function
            ew = 0
            for n in range(self.mini_batch):
                for k in range(k):
                    if y[n][k] == 0:
                        print("zeroooo")
                    ew += t[n][k] * np.log(y[n][k])
            ew -= self.lr * np.sum(np.square(w2)) / 2

            # backpropagation
            delta2 = t - y
            delta1 = delta2.dot(w2.T) * z * (1 - z)

            w2 += self.lr * (z.T.dot(delta2) - self.lr * w2)
            b2 += self.lr * delta2.sum(axis=0)

            w1 += self.lr * (x.T.dot(delta1) - self.lr * w1)
            b1 += self.lr * delta1.sum(axis=0)

            if i % 50 == 0:
                print('Loss function value: ', ew)
                costs.append(ew)

        return costs, w1, w2, b1, b2

    def predict(self, w1, w2, b1, b2, x_test):
        z_test = self.h(x_test.dot(w1) + b1)
        y_test = self.softmax(z_test.dot(w2) + b2)
        # Hard classification decisions
        t_test = np.argmax(y_test, 1)
        return t_test

    def create_batch(self, x, y):

        random_indices = np.random.choice(x.shape[0], size=self.mini_batch, replace=False)
        x_mini = x[random_indices, :]
        y_mini = y[random_indices, :]

        return x_mini, y_mini

    @staticmethod
    def softmax(x, ax=1):
        m = np.max(x, axis=ax, keepdims=True)  # max per row
        p = np.exp(x - m)

        return p / np.sum(p, axis=ax, keepdims=True)

    @staticmethod
    def h(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def h_derivative(z):
        return z * (1 - z)
