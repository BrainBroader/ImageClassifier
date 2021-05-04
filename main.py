from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    # plot 5 random images from the training set
    n = 100
    sqrt_n = int(n ** 0.5)
    samples = np.random.randint(X_train.shape[0], size=n)

    plt.figure(figsize=(11, 11))

    cnt = 0
    for i in samples:
        cnt += 1
        plt.subplot(sqrt_n, sqrt_n, cnt)
        plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')

    plt.show()
