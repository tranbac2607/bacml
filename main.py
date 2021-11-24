from utils import data
from learning.regression.linear import Model1


def main():
    x, y = data.read_csv('data/input.csv')
    x_train, y_train = x[:24], y[:24]
    x_test, y_test = x[24:], y[24:]
    model = Model1(x_train, y_train)
    model.train()
    print(model.predict(x_test))
    print(y_test)

if __name__ == '__main__':
    main()
