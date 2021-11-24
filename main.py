from utils import data
from learning.regression import linear
import sys


def main(filename):
    x, y = data.read_csv(filename)
    x_train, y_train = x[:24], y[:24]
    x_test, y_test = x[24:], y[24:]
    w = linear.train(x_train, y_train)
    print(w)
    print('---')
    print(linear.predict(x_test, w))
    print(y_test)

if __name__ == '__main__':
    if (len(sys.argv) > 1): main(sys.argv[1])
    else: print('Please enter file path!')