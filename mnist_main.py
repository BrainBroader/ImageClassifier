from sklearn.model_selection import train_test_split
from load_data import load_mnist
from sga_algorithm import SgaAlgorithm
import numpy as np

if __name__ == '__main__':
    train_path = 'ml/train%d.txt'  # sys.argv[1] + '\\train%d.txt'
    test_path = 'ml/test%d.txt'  # sys.argv[1] + '\\test%d.txt'

    # load training data
    print(f'[INFO] - Loading training data from {train_path}')
    X_train, y_train = load_mnist(train_path)
    print(f'[INFO] - Total train data: {len(X_train)}')

    print(f'[INFO] - Loading testing data from {test_path}')
    X_test, y_test = load_mnist(test_path)

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

    # sga = SgaAlgorithm()
    # weights = sga.train(train_data, train_target)
    # pred = sga.predict(weights, train_data)
    # print(np.mean(pred == np.argmax(train_target, 1)))
    for function in ["softplus", "cosine"]:
        for iterations in [200, 500]:
            for batch in [100, 200, 400]:
                for hidden_layer in [100, 200, 300, 700]:
                    for lr in [0.01, 0.03, 1e-3, 1e-4]:
                        for lamda in [1e-2, 1e-3, 1e-4]:

                            sga = SgaAlgorithm(function, lamda, lr, hidden_layer, batch, iterations)
                            weights = sga.train(train_data, train_target)
                            pred_train = sga.predict(weights, train_data)
                            pred_dev = sga.predict(weights, dev_data)
                            print("funxtion: " + function + " mini batch: " + str(batch) + " m: " + str(hidden_layer)
                                  + " lr: " + str(lr) + " lamda : " + str(lamda) + " iterations: " + str(iterations))
                            print("Accuracy for train: ")
                            print(np.mean(pred_train == np.argmax(train_target, 1)))
                            print("Accuracy for test: ")
                            print(np.mean(pred_dev == np.argmax(dev_target, 1)))
                            print("-----------------------------------")

    # n, d = train_data.shape
    # k = train_target.shape[1]
    #
    # weights = {"w1": np.random.randn(d, 100),
    #            "w2": np.random.randn(100, k),
    #            "b1": np.random.randn(100),
    #            "b2": np.random.randn(k)}
    #
    # gradEw, numericalGrad = sga.gradient_check(train_data, train_target, weights)
    # print(numericalGrad["b1"].shape)
    # print(gradEw["b1"].shape)
    #
    # # Absolute norm
    # print("The difference estimate for gradient of w is : ", np.max(np.abs(gradEw["w1"] - numericalGrad["w1"])))
