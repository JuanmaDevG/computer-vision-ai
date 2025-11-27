import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import keras

# First the type, then the constructor
from numpy import ndarray
from numpy import array

import pandas


def load_cifar10() -> tuple[ndarray]:
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    print(X_train.shape, X_train.dtype)
    print(Y_train.shape, Y_train.dtype)
    print(X_test.shape, X_test.dtype)
    print(Y_test.shape, Y_test.dtype)
    return X_train, Y_train, X_test, Y_test


def draw_random_image(X_sample: ndarray, Y_sample: ndarray):
    from random import randrange
    idx = randrange(0, len(X_sample))
    title = "Showing image X_sample[" + str(idx) + "] -- Y_sample[" + str(idx) + "] = " + str(Y_sample[idx])
    img = X_sample[idx]

    plt.figure()
    plt.suptitle(title)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_cifar10()

    draw_random_image(X_train, Y_train)

    # Tarea A
    #MLP = funcionMLP1(...)
    #MLP = funcionMLP2(...)

    # Tarea B
    #CNN = funcionCNN1(...)
    #CNN = funcionCNN2(...)

    # NOTA: dejar todo comentado menos la ultima tarea
