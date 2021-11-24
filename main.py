from utils import data
from learning.regression import linear


def main():
    x, y = data.read_csv('data/input.csv')
    x_train, y_train = x[:24], y[:24]
    x_test, y_test = x[24:], y[24:]
    w = linear.train(x_train, y_train)
    print(w)
    print('---')
    print(linear.predict(x_test, w))
    print(y_test)

if __name__ == '__main__':
    main()
