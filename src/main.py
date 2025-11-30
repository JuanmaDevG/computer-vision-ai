import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Tuple, List, Optional, Any, Dict

import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import regularizers

import numpy as np
import pandas
import time

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


###########################################################################################################
# Tool testing functions
###########################################################################################################
def loadprint_cifar10() -> tuple[np.ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    print(X_train.shape, X_train.dtype)
    print(Y_train.shape, Y_train.dtype)
    print(X_test.shape, X_test.dtype)
    print(Y_test.shape, Y_test.dtype)
    return X_train, Y_train, X_test, Y_test


def draw_random_image(X_sample: np.ndarray, Y_sample: np.ndarray):
    from random import randrange
    idx = randrange(0, len(X_sample))
    title = "Showing image X_sample[" + str(idx) + "] -- Y_sample[" + str(idx) + "] = " + str(Y_sample[idx])
    img = X_sample[idx]

    plt.figure()
    plt.suptitle(title)
    plt.imshow(img)
    plt.show()


def plot_sample_types(Y_sample: np.ndarray, xscale = 'linear', yscale = 'linear'):
    plt.title("Etiquetas de una muestra de " + str(len(Y_sample)) + " valores")
    plt.plot(Y_sample)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()
###########################################################################################################
###########################################################################################################


def preprocess_cifar10_mlp() -> tuple[np.ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test  = X_test.reshape((X_test.shape[0], -1))

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test  = to_categorical(Y_test, num_classes=10)
    return X_train, Y_train, X_test, Y_test

def preprocess_cifar10_cnn() -> tuple[np.ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    #TODO: preprocessing work here
    return X_train, Y_train, X_test, Y_test


def gfx_loss_evolution_and_success_rate(history: Dict[str, List[float]], title: str = "Evolución entrenamiento", filename='default.png'):
    h = history
    epochs = range(1, len(h['loss']) + 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, h['loss'], label='train_loss', linestyle='-')
    if 'val_loss' in h:
        ax1.plot(epochs, h['val_loss'], label='val_loss', linestyle='--')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    if 'accuracy' in h:
        ax2.plot(epochs, h['accuracy'], label='train_acc', linestyle='-')
    if 'val_accuracy' in h:
        ax2.plot(epochs, h['val_accuracy'], label='val_acc', linestyle='--')
    ax2.legend(loc='lower right')

    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches='tight')


def gfx_bars_training_time_and_final_success_rate(history: Dict[str, List[float]], title: str = "Evolución entrenamiento", filename='default.png'):
    labels = [r['label'] for r in results]
    times  = [r['time'] for r in results]
    accs   = [r['test_acc'] for r in results]

    x = np.arange(len(labels))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.bar(x, times, width)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempos de entrenamiento')

    ax2.bar(x, accs, width)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Test accuracy')
    ax2.set_title('Accuracies finales (test)')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')


def gfx_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None, title: str = "Matriz de confusión"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches='tight')


def make_new_mlp(input_dim: int,
                  hidden_layers: List[int],
                  activation: str = 'sigmoid',
                  output_units: int = 10,
                  output_activation: str = 'softmax',
                  kernel_initializer: str = 'glorot_uniform',
                  l2_reg: float = 0.0,
                  dropout: float = 0.0,
                  use_batchnorm: bool = False) -> keras.models.Model:
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for i, units in enumerate(hidden_layers):
        if l2_reg > 0:
            reg = regularizers.l2(l2_reg)
        else:
            reg = None
        model.add(layers.Dense(units,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=reg,
                               name=f"dense_{i+1}"))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
        if dropout and dropout > 0.0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_units, activation=output_activation))
    return model


def try_mlp(X_train: np.ndarray, Y_train: np.ndarray,
               X_test: np.ndarray, Y_test: np.ndarray,
               hidden_layers: List[int] = [48],
               activation: str = 'sigmoid',
               kernel_initializer: str = 'glorot_uniform',
               batch_size: int = 32,
               epochs: int = 10,
               use_earlystopping: bool = True,
               patience: int = 5,
               l2_reg: float = 0.0,
               dropout: float = 0.0,
               use_batchnorm: bool = False,
               repetitions: int = 1,
               verbose: int = 1
               ) -> Dict[str, Any]:
    results = {
        'models': [],
        'histories': [],
        'test_scores': [],
        'times': []
    }

    for rep in range(repetitions):
        model = make_new_mlp(input_dim=X_train.shape[1],
                              hidden_layers=hidden_layers,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              l2_reg=l2_reg,
                              dropout=dropout,
                              use_batchnorm=use_batchnorm)
        model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        if verbose:
            model.summary()

        cb = []
        if use_earlystopping:
            cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0))

        t0 = time.time()
        history = model.fit(X_train, Y_train, validation_split=0.1, epochs=epochs,
                            batch_size=batch_size, callbacks=cb, verbose=verbose)
        t1 = time.time()
        elapsed = t1 - t0

        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

        results['models'].append(model)
        results['histories'].append(history.history)
        results['test_scores'].append((test_loss, test_acc))
        results['times'].append(elapsed)

    avg_loss = float(np.mean([s[0] for s in results['test_scores']]))
    avg_acc  = float(np.mean([s[1] for s in results['test_scores']]))
    avg_time = float(np.mean(results['times']))

    summary = {
        'models': results['models'],
        'histories': results['histories'],
        'avg_test_loss': avg_loss,
        'avg_test_acc': avg_acc,
        'avg_time': avg_time,
        'raw': results
    }
    return summary


####################################
# Tareas expuestas en el enunciado #
####################################

def tareaMLP1() -> keras.models.Model:
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    model = keras.models.Sequential([
        layers.InputLayer(shape=(X_train.shape[1],)),
        layers.Dense(48, activation='sigmoid'),
        layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 10% validation split to watch loss and accuracy evolution per epoch
    t = time.time()
    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=10).history
    t = time.time() - t
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    data = {
            'time': t,
            'accuracy': history['accuracy'],
            'loss': history['loss'],
            'val_accuracy': history['val_accuracy'],
            'val_loss': history['val_loss'],
            'avg_loss': test_loss,
            'avg_accuracy': test_accuracy
            }

    epochs = range(1, len(history['loss']) +1)
    plt.title('Tarea MLP 1: graphically comparing attribtues')
    fig, ax_acc = plt.subplots()

    fig.text(0.05, 1.0, f"Evaluated accuracy: {data['avg_accuracy'] * 100:.0f}%", ha='left', va='top', fontsize=12, color='green')
    fig.text(0.05, 0.95, f"Evaluated loss: {data['avg_loss']:.2f}", ha='left', va='top', fontsize=12, color='red')

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.plot(epochs, data['accuracy'], label='Trainig accuracy', color='green')
    ax_acc.plot(epochs, data['val_accuracy'], label='Validation accuracy', color='green', linestyle='--')
    ax_acc.legend(loc='lower left')

    ax_loss = ax_acc.twinx()
    ax_loss.set_ylabel('Loss')
    ax_loss.plot(epochs, data['loss'], label='Training loss', color='red')
    ax_loss.plot(epochs, data['val_loss'], label='Validation loss', color='red', linestyle='--')
    ax_loss.legend(loc='upper left')

    plt.savefig('MLP_tarea1.png', dpi=150, bbox_inches='tight')
    return model


def tareaMLP2(epochs_list: List[int] = [5, 10, 20], repetitions: int = 3):
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for e in epochs_list:
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=[48],
                         activation='sigmoid',
                         epochs=e,
                         use_earlystopping=True,
                         patience=5,
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': f"epochs_{e}", 'time': res['avg_time'], 'test_acc': res['avg_test_acc'], 'histories': res['histories']})
        gfx_loss_evolution_and_success_rate(res['histories'][0], title=f"epochs={e}", filename=f"MLP_tarea2_epoch_{e}.png")
    gfx_bars_training_time_and_final_success_rate(results, title="MLP - ajuste epochs", filename='MLP_tarea2_ajuste_epochs.png')
    return results


def tareaMLP3(batch_sizes: List[int] = [16, 32, 64], repetitions: int = 3):
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for b in batch_sizes:
        print(f"\nRight now batch_size = {b}")
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=[48],
                         activation='sigmoid',
                         batch_size=b,
                         epochs=20,
                         use_earlystopping=True,
                         patience=5,
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': f"bs_{b}", 'time': res['avg_time'], 'test_acc': res['avg_test_acc']})
    gfx_bars_training_time_and_final_success_rate(results, title="MLP - ajuste batch_size", filename='MLP_tarea3_ajuste_batch_size.png')
    return results


def tareaMLP4(activations_and_inits: Optional[List[Dict[str, str]]] = None, repetitions: int = 3):
    if activations_and_inits is None:
        activations_and_inits = [
            {'activation': 'sigmoid', 'initializer': 'glorot_uniform'},
            {'activation': 'tanh',    'initializer': 'glorot_uniform'},
            {'activation': 'relu',    'initializer': 'he_normal'},
            {'activation': 'elu',     'initializer': 'he_normal'},
        ]
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for cfg in activations_and_inits:
        label = f"{cfg['activation']}_{cfg['initializer']}"
        print(f"\nNow trying {label}")
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=[48],
                         activation=cfg['activation'],
                         kernel_initializer=cfg['initializer'],
                         epochs=20,
                         use_earlystopping=True,
                         patience=5,
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': label, 'time': res['avg_time'], 'test_acc': res['avg_test_acc'], 'histories': res['histories']})
    gfx_bars_training_time_and_final_success_rate(results, title="MLP - activacion / inicializador", filename='MLP_tarea4_activacion_inicializador.png')
    return results


def tareaMLP5(neuron_counts: List[int] = [16, 32, 48, 64, 96], repetitions: int = 3):
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for n in neuron_counts:
        print(f"\nTrying {n} neurons")
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=[n],
                         activation='relu',
                         kernel_initializer='he_normal',
                         epochs=20,
                         use_earlystopping=True,
                         patience=5,
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': f"{n}n", 'time': res['avg_time'], 'test_acc': res['avg_test_acc']})
    gfx_bars_training_time_and_final_success_rate(results, title="MLP - ajuste nº neuronas", filename='MLP_tarea5_ajuste_de_neuronas.png')
    return results


def tareaMLP6(configs: Optional[List[List[int]]] = None, repetitions: int = 3):
    if configs is None:
        configs = [[96], [48,48], [32,64], [64,32], [32,32,32], [16,32,48]]
    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for cfg in configs:
        label = "+".join(map(str, cfg))
        print(f"\nProbando capas {label}")
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=cfg,
                         activation='relu',
                         kernel_initializer='he_normal',
                         epochs=30,
                         use_earlystopping=True,
                         patience=6,
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': label, 'time': res['avg_time'], 'test_acc': res['avg_test_acc']})
    gfx_bars_training_time_and_final_success_rate(results, title="MLP - capas y distribución neuronas", filename='MLP_tarea6_capas_y_neuronas.png')
    return results


def tareaMLP7(repetitions: int = 3):
    candidates = [
        {'hidden': [128, 64], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.3, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [256, 128], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.4, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [96, 48, 32], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.2, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [512], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.5, 'l2': 1e-3, 'batchnorm': False},
        {'hidden': [384, 256], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.4, 'l2': 1e-4, 'batchnorm': True},
    ]
    # Ensure less than 1000 neurons
    candidates = [c for c in candidates if sum(c['hidden']) <= 1000 and len(c['hidden']) <= 6]

    X_train, Y_train, X_test, Y_test = preprocess_cifar10_mlp()
    results = []
    for c in candidates:
        label = f"{'+'.join(map(str,c['hidden']))}_act-{c['activation']}_do-{c['dropout']}_l2-{c['l2']}"
        print(f"\nProbando candidate: {label}")
        res = try_mlp(X_train, Y_train, X_test, Y_test,
                         hidden_layers=c['hidden'],
                         activation=c['activation'],
                         kernel_initializer=c['init'],
                         batch_size=64,
                         epochs=50,
                         use_earlystopping=True,
                         patience=6,
                         l2_reg=c['l2'],
                         dropout=c['dropout'],
                         use_batchnorm=c['batchnorm'],
                         repetitions=repetitions,
                         verbose=0)
        results.append({'label': label, 'time': res['avg_time'], 'test_acc': res['avg_test_acc'], 'raw': res})

    results = sorted(results, key=lambda r: r['test_acc'], reverse=True)
    gfx_bars_training_time_and_final_success_rate(results, title="MLP7 - Optimización final", filename='MLP_tarea7_optimizacion_final.png')

    best = results[0]
    best_model = best['raw']['models'][0]

    _, _, X_test_raw, Y_test_raw = None, None, None, None
    X_train_p, Y_train_p, X_test_p, Y_test_p = preprocess_cifar10_mlp()
    y_true = np.argmax(Y_test_p, axis=1)
    y_pred = np.argmax(best_model.predict(X_test_p), axis=1)
    matriz_de_confusion(y_true, y_pred, labels=[str(i) for i in range(10)], title=f"Matriz confusión - {best['label']}")
    print("Top candidates (label, acc, time):")
    for r in results[:5]:
        print(r['label'], r['test_acc'], f"{r['time']:.2f}s")
    return results


if __name__ == "__main__":
    # Tool testing (uncomment to use)
    ############################################################
    #(X_train, Y_train), (X_test, Y_test) = loadprint_cifar10()
    #draw_random_image(X_train, Y_train)
    #plot_sample_types(Y_test[:20])
    ############################################################

    # Primera parte, tareas con MLP
    MLP = tareaMLP1()
    #MLP = tareaMLP2()
    #MLP = tareaMLP3()
    #MLP = tareaMLP4()
    #MLP = tareaMLP5()
    #MLP = tareaMLP6()
    #MLP = tareaMLP7()

    # Tarea B
    #CNN = funcionCNN1(...)
    #CNN = funcionCNN29(...)

    # NOTA: dejar todo comentado menos la ultima tarea
