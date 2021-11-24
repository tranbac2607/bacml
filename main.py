from utils import data
from learning.regression.linear import Model1
import sys


def main(filename):
    x, y = data.read_csv(filename)
    x_train, y_train = x[:24], y[:24]
    x_test, y_test = x[24:], y[24:]
    model = Model1(x_train, y_train)
    model.train()
    print(model.predict(x_test))
    print(y_test)

if __name__ == '__main__':
    if (len(sys.argv) > 1): main(sys.argv[1])
    else: print('Please enter file path!')