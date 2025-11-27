import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import cifar10

# First the type, then the constructor
from numpy import ndarray
from numpy import array

import pandas


###########################################################################################################
# Tool testing functions
###########################################################################################################
def loadprint_cifar10() -> tuple[ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
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


def plot_sample_types(Y_sample: ndarray, xscale = 'linear', yscale = 'linear'):
    plt.title("Etiquetas de una muestra de " + str(len(Y_sample)) + " valores")
    plt.plot(Y_sample)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()
###########################################################################################################
###########################################################################################################


def load_cifar10_mlp() -> tuple[ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    #TODO: preprocessing work here
    return X_train, Y_train, X_test, Y_test

def load_cifar10_cnn() -> tuple[ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    #TODO: preprocessing work here
    return X_train, Y_train, X_test, Y_test


#TODO: functions when possible
def grafica_de_lineas_con_evolucion_de_perdida_y_tasa_de_acierto():
    pass
def grafica_de_barras_con_tiempo_de_entrenamiento_y_tasas_de_acierto_finales():
    pass
def matriz_de_confusion():
    pass


if __name__ == "__main__":
    # Tool testing (uncomment to use)
    ############################################################
    #(X_train, Y_train), (X_test, Y_test) = loadprint_cifar10()
    #draw_random_image(X_train, Y_train)
    #plot_sample_types(Y_test[:20])
    ############################################################

    (X_train, Y_train), (X_test, Y_test) = load_cifar10_mlp()

    # Primera parte, tareas con MLP
    MLP = tareaMLP1(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP2(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP3(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP4(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP5(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP6(X_train, Y_train, X_test, Y_test)
    #MLP = tareaMLP7(X_train, Y_train, X_test, Y_test)

    # Tarea B
    #CNN = funcionCNN1(...)
    #CNN = funcionCNN2(...)

    # NOTA: dejar todo comentado menos la ultima tarea
