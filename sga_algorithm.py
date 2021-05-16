import numpy as np


class SgaAlgorithm:

    def __init__(self, function="tanh", lamda=0.001, lr=0.001, m=200, mini_batch=100, iterations=500, tol=1e-6):
        self.function = function
        self.m = m
        self.mini_batch = mini_batch
        self.iterations = iterations
        self.lamda = lamda
        self.tol = tol
        self.lr = lr

    def train(self, x, t):
        """
        It trains the model, initializes the parameters of the model and the for a number of iterations
        calls the method calculate_costs and upgrades the value of the parameters.

        :param x: 2D np array that contains the train dataset
        :param t: 2D np array that shows the category of each image

        :return: the parameter's values of the trained model
        """

        costs = []
        e_wold = -np.inf

        n, d = x.shape
        k = t.shape[1]

        # random initialization of weights and bias
        weights = {"w1": np.random.randn(d, self.m),
                   "w2": np.random.randn(self.m, k),
                   "b1": np.random.randn(self.m),
                   "b2": np.random.randn(k)}

        for i in range(self.iterations):

            x, t = self.create_batch(x, t)

            ew, gradients = self.calculate_cost(x, t, weights)

            costs.append(ew)
            # Show the current cost function on screen
            if i % 100 == 0:
                print('Iteration : %d, Cost function :%f' % (i, ew))

            # Break if you achieve the desired accuracy in the cost function
            if np.abs(ew - e_wold) < self.tol:
                break

            # update weights
            weights["w2"] += self.lr * gradients["w2"]
            weights["b2"] += self.lr * gradients["b2"]

            weights["w1"] += self.lr * gradients["w1"]
            weights["b1"] += self.lr * gradients["b1"]

        return weights

    def calculate_cost(self, x, t, weights):

        """
        Performs forward, back propagation and calculates the cost function.

        :param x: 2D np array that contains the train dataset
        :param t: 2D np array that shows the category of each image
        :param weights: the parameters of the model

        :return: ew: the value of the cost
                 gradients: dictionary that contains the gradients of each parameter
        """

        gradients = {}
        # forward step
        # phase 1

        z = np.dot(x, weights["w1"]) + weights["b1"]
        hz = self.act_function(z, self.function)

        # phase 2
        y = np.dot(hz, weights["w2"]) + weights["b2"]
        ynk = self.softmax(y)

        m = np.max(y, axis=1)
        ew = np.sum(t * y) - np.sum(m) -\
            np.sum(np.log(np.sum(np.exp(y - np.expand_dims(m, axis=1)), 1))) -\
            (self.lamda / 2) * (np.sum(np.square(weights["w1"])) + np.sum(np.square(weights["w2"])))

        # back propagation
        delta2 = t - ynk
        delta1 = np.dot(delta2, weights["w2"].T) * self.act_function_der(z, self.function)

        # phase 1
        gradients["w2"] = np.dot(hz.T, delta2) - self.lamda * weights["w2"]
        gradients["b2"] = delta2.sum(axis=0)

        # phase 2
        gradients["w1"] = np.dot(x.T, delta1) - self.lamda * weights["w1"]
        gradients["b1"] = delta1.sum(axis=0)

        return ew, gradients

    def predict(self, weights, x_test):
        z_test = self.act_function(x_test.dot(weights["w1"]) + weights["b1"], self.function)
        y_test = self.softmax(z_test.dot(weights["w2"]) + weights["b2"])
        # Hard classification decisions
        t_test = np.argmax(y_test, 1)
        return t_test

    def create_batch(self, x, y):
        """
            Chooses randomly a number of indices/train examples without replace
            and create a new batch of train examples.

        :param x: 2D np array that contains the train dataset
        :param y: 2D np array that shows the category of each image
        :return: x_mini: mini batch of array x
                 y_mini: mini batch of array y
        """

        random_indices = np.random.choice(x.shape[0], size=self.mini_batch, replace=False)
        x_mini = x[random_indices, :]
        y_mini = y[random_indices, :]

        return x_mini, y_mini

    def gradient_check(self, x, t, weights):

        epsilon = 1e-7

        _list = np.random.randint(x.shape[0], size=5)
        x_sample = np.array(x[_list, :])
        t_sample = np.array(t[_list, :])

        ew, grad = self.calculate_cost(x_sample, t_sample, weights)

        numerical_grad = {"w1": np.zeros_like(grad["w1"]),
                          "w2": np.zeros_like(grad["w2"]),
                          "b1": np.zeros_like(grad["b1"]),
                          "b2": np.zeros_like(grad["b2"])}

        # Compute numerical_grad
        for i in weights:
            if i == "w1" or i == "w2":
                for k in range(grad[i].shape[0]):
                    for d in range(grad[i].shape[1]):

                        thetaplus = weights  # Step 1
                        thetaplus[i][k, d] += epsilon  # Step 2
                        e_plus, _ = self.calculate_cost(x_sample, t_sample, thetaplus)  # Step 3

                        thetaminus = weights  # Step 1
                        thetaminus[i][k, d] -= epsilon  # Step 2
                        e_minus, _ = self.calculate_cost(x_sample, t_sample, thetaminus)  # Step 3

                        numerical_grad[i][k, d] = (e_plus - e_minus) / (2 * epsilon)
            else:
                for k in range(grad[i].shape[0]):
                    thetaplus = weights  # Step 1
                    thetaplus[i][k] += epsilon  # Step 2
                    e_plus, _ = self.calculate_cost(x_sample, t_sample, thetaplus)  # Step 3

                    thetaminus = weights  # Step 1
                    thetaminus[i][k] -= epsilon  # Step 2
                    e_minus, _ = self.calculate_cost(x_sample, t_sample, thetaminus)  # Step 3

                    numerical_grad[i][k] = (e_plus - e_minus) / (2 * epsilon)

        return grad, numerical_grad

    @staticmethod
    def softmax(x, ax=1):
        m = np.max(x, axis=ax, keepdims=True)  # max per row
        p = np.exp(x - m)

        return p / np.sum(p, axis=ax, keepdims=True)

    @staticmethod
    def act_function(z, function):
        if function == "cosine":
            return np.cos(z)
        elif function == "softplus":
            return np.log(1 + np.exp(z))
        else:
            return np.tanh(z)

    @staticmethod
    def act_function_der(z, function):
        if function == "cosine":
            return - np.sin(z)
        elif function == "softplus":
            return np.exp(z) / (1 + np.exp(z))
        else:
            return 1 - np.square(np.tanh(z))
