import numpy as np

def train(x, y, iterations=100, learning_rate=0.00001):
    n = x.shape[0]
    X = np.hstack([np.ones((n, 1)), x])
    w = np.array([0., 1.]).reshape(-1, 1)
    for iter in range(iterations):
        z = np.dot(X, w)
        loss = z - y
        cost = 0.5 * np.sum(loss * loss)
        w[0] -= learning_rate * np.sum(loss)
        w[1] -= learning_rate * np.sum(np.multiply(loss, X[:,1].reshape(-1,1)))
    return w

def predict(x, w):
    return w[1] * x + w[0]

