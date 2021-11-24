import numpy as np


class Model:
    def train(self, iterations, learning_rate):
        pass

    def predict(self, x):
        pass


class Model1(Model):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self, iterations=100, learning_rate=0.00001):
        x, y = self.x_train, self.y_train
        n = x.shape[0]
        X = np.hstack([np.ones((n, 1)), x])
        w = np.array([0., 1.]).reshape(-1, 1)
        for iter in range(iterations):
            z = np.dot(X, w)
            loss = z - y
            cost = 0.5 * np.sum(loss * loss)
            w[0] -= learning_rate * np.sum(loss)
            w[1] -= learning_rate * np.sum(np.multiply(loss, X[:,1].reshape(-1,1)))
        self.w = w

    def predict(self, x):
        w = self.w
        return w[1] * x + w[0]

