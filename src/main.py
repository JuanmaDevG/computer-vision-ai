import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

import logging, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def save_figure(fig, filename: str):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensure_reproducibility(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_preprocess_mlp() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    return X_train, Y_train, X_test, Y_test


def load_preprocess_cnn() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pass


def build_mlp(input_dim: int,
              output_units: int = 10,
              hidden_layers: List[int] = [48],
              activation: str = "sigmoid",
              kernel_initializer: str = "glorot_uniform",
              l2_reg: float = 0.0,
              dropout: float = 0.0,
              use_batchnorm: bool = False,
              out_activation: str = "softmax") -> keras.models.Model:
    model = keras.models.Sequential()
    model.add(layers.InputLayer(shape=(input_dim,)))
    for units in hidden_layers:
        reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
        model.add(layers.Dense(units, activation=activation,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=reg))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
        if dropout and dropout > 0.0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_units, activation=out_activation))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_evaluate(model: keras.models.Model,
                       X_train: np.ndarray, Y_train: np.ndarray,
                       X_test: np.ndarray, Y_test: np.ndarray,
                       epochs: int = 10,
                       batch_size: int = 32,
                       validation_split: float = 0.1,
                       use_earlystopping: bool = False,
                       es_patience: int = 5,
                       verbose: int = 1) -> Dict[str, Any]:
    callbacks = []
    if use_earlystopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss",
                                                       patience=es_patience,
                                                       restore_best_weights=True,
                                                       verbose=0))

    t0 = time.time()
    history = model.fit(X_train, Y_train,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=verbose)
    train_time = time.time() - t0
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

    return {
        "model": model,
        "history": history.history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time": train_time
    }


def plot_loss_acc(history: Dict[str, List[float]], title: str, filename: str):
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, history["loss"], label="train_loss", linestyle="-")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="val_loss", linestyle="--")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    if "accuracy" in history:
        ax2.plot(epochs, history["accuracy"], label="train_acc", linestyle="-")
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], label="val_acc", linestyle="--")
    ax2.legend(loc="lower right")

    plt.title(title)
    save_figure(fig, filename)


def plot_bars_results(results: List[Dict[str, Any]], title: str, filename: str):
    labels = [r["label"] for r in results]
    times = [r["train_time"] for r in results]
    accs = [r["test_acc"] for r in results]

    x = np.arange(len(labels))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.bar(x, times, width)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Tiempo (s)")
    ax1.set_title("Tiempos de entrenamiento")

    ax2.bar(x, accs, width)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Accuracies finales (test)")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, filename)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]], title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation="vertical", cmap=plt.cm.Blues)
    plt.title(title)
    save_figure(fig, filename)



def tareaMLP1() -> keras.models.Model:
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    model = build_mlp(input_dim=X_train.shape[1],
                      hidden_layers=[48],
                      activation="sigmoid",
                      kernel_initializer="glorot_uniform",
                      out_activation="softmax")
    model.summary()

    res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)

    plot_loss_acc(res["history"], title="MLP1 - evolución (48 neurons, sigmoid, 10 epochs)",
                  filename="MLP_tarea1_evolucion.png")

    print(f"MLP1 - test_loss: {res['test_loss']:.4f}, test_acc: {res['test_acc']:.4f}, train_time: {res['train_time']:.1f}s")
    return res["model"]


def tareaMLP2(epochs_list: List[int] = [1, 5, 10, 25, 40, 70], repetitions: int = 5, use_earlystopping: bool = False):
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []

    for e in epochs_list:
        label = f"epochs_{e}"
        accs = []
        times = []
        histories = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=42 + rep)
            model = build_mlp(input_dim=X_train.shape[1], hidden_layers=[48], activation="sigmoid")
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test, epochs=e,
                                     use_earlystopping=use_earlystopping, es_patience=5, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            histories.append(res["history"])

        results.append({
            "label": label,
            "test_acc": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "train_time": float(np.mean(times)),
            "histories": histories
        })
        print(f"{label}: mean_acc={np.mean(accs):.4f} (std={np.std(accs):.4f}), mean_time={np.mean(times):.1f}s")

    plot_bars_results(results, title="MLP2 - Comparativa epochs", filename="MLP_tarea2_epochs_comparativa.png")
    for r in results:
        hist0 = r["histories"][0]
        plot_loss_acc(hist0, title=f"MLP epochs {r['label']}", filename=f"MLP_tarea2_{r['label']}_evol.png")

    return results


def tareaMLP3(batch_sizes: List[int] = [8, 16, 32, 64, 128, 256], repetitions: int = 3, activation = "sigmoid"):
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for b in batch_sizes:
        label = f"bs_{b}"
        accs = []
        times = []
        histories = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=123 + rep)
            model = build_mlp(input_dim=X_train.shape[1], hidden_layers=[48], activation=activation, kernel_initializer="he_normal")
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=20, batch_size=b, validation_split=0.1, es_patience=5, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            histories.append(res["history"])
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times)), "histories": histories})
        print(f"{label}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")

    plot_bars_results(results, title="MLP3 - Comparativa batch_size", filename="MLP_tarea3_batchsize.png")
    return results


def tareaMLP4(activations_and_inits: Optional[List[Dict[str, str]]] = None,
              repetitions: int = 3,
              use_earlystopping: bool = False):
    if activations_and_inits is None:
        activations_and_inits = [
            {'activation': 'sigmoid', 'initializer': 'glorot_uniform'},
            {'activation': 'tanh',    'initializer': 'glorot_uniform'},
            {'activation': 'relu',    'initializer': 'he_normal'},
            {'activation': 'elu',     'initializer': 'he_normal'},
        ]
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for cfg in activations_and_inits:
        label = f"{cfg['activation']}_{cfg['initializer']}"
        accs = []
        times = []
        histories = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=7 + rep)
            model = build_mlp(input_dim=X_train.shape[1],
                              hidden_layers=[48],
                              activation=cfg['activation'],
                              kernel_initializer=cfg['initializer'])
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=20, batch_size=32, validation_split=0.1,
                                     use_earlystopping=use_earlystopping, es_patience=5, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            histories.append(res["history"])
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times)), "histories": histories})
        print(f"{label}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")

    plot_bars_results(results, title="MLP4 - activaciones / inicializadores", filename="MLP_tarea4_activ_init.png")
    return results


def tareaMLP5(neuron_counts: List[int] = [16, 32, 48, 64, 96, 128, 256, 512, 1024], repetitions: int = 3, use_earlystopping: bool = False):
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for n in neuron_counts:
        label = f"{n}n"
        accs = []
        times = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=50 + rep)
            model = build_mlp(input_dim=X_train.shape[1], hidden_layers=[n], activation="relu", kernel_initializer="he_normal")
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=20, batch_size=32, validation_split=0.1,
                                     use_earlystopping=use_earlystopping, es_patience=5, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times))})
        print(f"{label}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")

    plot_bars_results(results, title="MLP5 - ajuste nº neuronas", filename="MLP_tarea5_neuronas.png")
    return results


def tareaMLP6(configs: Optional[List[List[int]]] = None, repetitions: int = 3, use_earlystopping: bool = False):
    if configs is None:
        configs = [[96], [48, 48], [32, 64], [64, 32], [32, 32, 32], [16, 32, 48], [96, 64, 48, 32, 16]]
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for cfg in configs:
        label = "+".join(map(str, cfg))
        accs = []
        times = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=200 + rep)
            model = build_mlp(input_dim=X_train.shape[1], hidden_layers=cfg, activation="relu", kernel_initializer="he_normal")
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=30, batch_size=32, validation_split=0.1,
                                     use_earlystopping=use_earlystopping, es_patience=6, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times))})
        print(f"{label}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")
    plot_bars_results(results, title="MLP6 - capas y distribución de neuronas", filename="MLP_tarea6_capas_neuronas.png")
    return results


def tareaMLP7(repetitions: int = 10, use_earlystopping: bool = True):
    #RECUERDA: menos de 1000 neuronas y maximo de 6 capas
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    INPUT_DIM = X_train.shape[1]
    OUTPUT_UNITS = 10

    class BuildParams(NamedTuple):
        hidden_layers: List[int]
        activation: str = 'sigmoid'
        kernel_initializer: str = 'glorot_uniform'
        l2_reg: float = 0.0
        dropout: float = 0.0
        batchnorm: bool = False
        out_activation: str = 'softmax'

    class TrainParams(NamedTuple):
        epochs: int
        batch_size: int
        validation_split: float = 0.1
        earlystopping: bool = False
        es_patience: int = 5
        verbose: int = 0

    class ModelTemplate(NamedTuple):
        buildparams: BuildParams
        trainparams: TrainParams
        const_buildparams: Tuple = (INPUT_DIM, OUTPUT_UNITS)

    candidates = [
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[128, 64],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-3,
                    dropout=0.3,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=16,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[128, 64],
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',
                    l2_reg=1e-4,
                    dropout=0.15,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=60,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=8
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[256, 128],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-4,
                    dropout=0.4,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[96, 64, 32],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-4,
                    dropout=0.25,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[512],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-3,
                    dropout=0.5,
                    batchnorm=False
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[64, 64],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-4,
                    dropout=0.2,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers=[64, 48, 32, 16],
                    activation='relu',
                    kernel_initializer='he_normal',
                    l2_reg=1e-4,
                    dropout=0.25,
                    batchnorm=True
                ),
                trainparams = TrainParams(
                    epochs=50,
                    batch_size=64,
                    earlystopping=True,
                    es_patience=6
                )
            ),
            ModelTemplate(
                buildparams = BuildParams(
                    hidden_layers = [512, 256, 64, 32, 16],
                    activation = 'relu', 
                    kernel_initializer = 'he_normal',
                    l2_reg=1e-4,
                    dropout=0.25,
                    batchnorm = False
                ),
                trainparams = TrainParams(
                    epochs = 30,
                    batch_size = 256,
                    earlystopping = True
                )
            )
        ]

    results = []
    for idx, c in enumerate(candidates):
        label = f"ModelTemplate_{idx}"
        info_tag = f"{'+'.join(map(str, c.buildparams.hidden_layers))}_act-{c.buildparams.activation}" \
                f"_do-{c.buildparams.dropout}_batchnorm-{c.buildparams.batchnorm}_l2-{c.buildparams.l2_reg}" \
                f"_epochs-{c.trainparams.epochs}_batchsize-{c.trainparams.batch_size}" \
                + (f"_es-patience-{c.trainparams.es_patience}" if c.trainparams.earlystopping else "")
        print(f"Probando candidate: {info_tag}")
        accs = []
        times = []
        raws = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=999 + rep)
            model = build_mlp(*c.const_buildparams, *c.buildparams)
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test, *c.trainparams)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            raws.append(res)
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times)), "raws": raws})
        print(f"{info_tag}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")

    results_sorted = sorted(results, key=lambda r: r["test_acc"], reverse=True)
    plot_bars_results(results_sorted, title="MLP7 - optimización final", filename="MLP_tarea7_optim_final.png")

    best = results_sorted[0]
    best_raw = best["raws"][0]
    best_model = best_raw["model"]

    X_train_p, Y_train_p, X_test_p, Y_test_p = load_preprocess_mlp()
    y_true = np.argmax(Y_test_p, axis=1)
    y_pred = np.argmax(best_model.predict(X_test_p), axis=1)
    labels = [str(i) for i in range(10)]
    plot_confusion_matrix(y_true, y_pred, labels=labels, title=f"Matriz confusión - {best['label']}", filename="MLP_tarea7_confmat_best.png")

    # imprimir top 5
    print("\nTop candidates (info_tag, acc, time):")
    for r in results_sorted[:5]:
        print(r["label"], f"{r['test_acc']:.4f}", f"{r['train_time']:.1f}s")
    return results_sorted


def load_preprocess_cnn() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test  = to_categorical(Y_test, num_classes=10)
    return X_train, Y_train, X_test, Y_test


def build_cnn(input_shape: Tuple[int, int, int] = (32, 32, 3),
              num_classes: int = 10,
              filters_block1: int = 16,
              filters_block2: int = 32,
              kernel_size: Tuple[int, int] = (3, 3),
              activation: str = "relu",
              kernel_initializer: str = "he_normal",
              dense_units: int = 100,
              use_batchnorm: bool = False,
              dropout: float = 0.0) -> keras.models.Model:
    model = keras.models.Sequential()
    model.add(layers.InputLayer(shape=input_shape))

    model.add(layers.Conv2D(filters_block1, kernel_size, padding="same",
                            kernel_initializer=kernel_initializer))
    if use_batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters_block2, kernel_size, padding="same",
                            kernel_initializer=kernel_initializer))
    if use_batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    if dropout and dropout > 0.0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation="relu", kernel_initializer="he_normal"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def tareaCNN1(epochs: int = 30,
              batch_size: int = 64,
              validation_split: float = 0.1,
              use_earlystopping: bool = True,
              es_patience: int = 6,
              verbose: int = 0) -> keras.models.Model:
    X_train, Y_train, X_test, Y_test = load_preprocess_cnn()
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=Y_train.shape[1],
                      filters_block1=16, filters_block2=32, kernel_size=(3, 3),
                      activation="relu", kernel_initializer="he_normal",
                      dense_units=100, use_batchnorm=False, dropout=0.0)
    model.summary()

    res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                             epochs=epochs, batch_size=batch_size,
                             validation_split=validation_split,
                             use_earlystopping=use_earlystopping,
                             es_patience=es_patience, verbose=verbose)

    plot_loss_acc(res["history"], title=f"CNN1 - evolución (16/32 filters, kernel 3x3)",
                  filename="CNN_tarea1_evolucion.png")
    print(f"CNN1 - test_loss: {res['test_loss']:.4f}, test_acc: {res['test_acc']:.4f}, train_time: {res['train_time']:.1f}s")
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(res["model"].predict(X_test), axis=1)
    labels = [str(i) for i in range(Y_test.shape[1])]
    plot_confusion_matrix(y_true, y_pred, labels=labels, title="CNN1 - Matriz de confusión",
                          filename="CNN_tarea1_confmat.png")

    return res["model"]


def tareaCNN2(kernel_sizes: Optional[List[Tuple[int, int]]] = None,
              repetitions: int = 3,
              epochs: int = 40,
              batch_size: int = 64,
              validation_split: float = 0.1,
              use_earlystopping: bool = True,
              es_patience: int = 6,
              verbose: int = 0):
    if kernel_sizes is None:
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]

    X_train, Y_train, X_test, Y_test = load_preprocess_cnn()
    results = []

    for k in kernel_sizes:
        label = f"kernel_{k[0]}x{k[1]}"
        accs = []
        times = []
        histories = []
        print(f"\nProbando kernel size: {k}")
        for rep in range(repetitions):
            ensure_reproducibility(seed=100 + rep)
            model = build_cnn(input_shape=X_train.shape[1:], num_classes=Y_train.shape[1],
                              filters_block1=16, filters_block2=32, kernel_size=k,
                              activation="relu", kernel_initializer="he_normal",
                              dense_units=100, use_batchnorm=False, dropout=0.0)

            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=epochs, batch_size=batch_size,
                                     validation_split=validation_split,
                                     use_earlystopping=use_earlystopping,
                                     es_patience=es_patience,
                                     verbose=verbose)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            histories.append(res["history"])

            if rep == 0:
                plot_loss_acc(res["history"], title=f"CNN - kernel {k[0]}x{k[1]} - evolución",
                              filename=f"CNN_tarea2_kernel_{k[0]}x{k[1]}_evol.png")

        results.append({
            "label": label,
            "test_acc": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "train_time": float(np.mean(times)),
            "histories": histories
        })
        print(f"{label}: mean_acc={np.mean(accs):.4f} (std={np.std(accs):.4f}), mean_time={np.mean(times):.1f}s")

    plot_bars_results(results, title="CNN2 - comparativa kernel sizes", filename="CNN_tarea2_kernels_comparativa.png")

    results_sorted = sorted(results, key=lambda r: r["test_acc"], reverse=True)
    best = results_sorted[0]
    best_kernel_label = best["label"]
    print(f"\nMejor kernel: {best_kernel_label} (acc = {best['test_acc']:.4f})")

    best_k = tuple(map(int, best["label"].split("_")[1].split("x")))
    final_model = build_cnn(input_shape=X_train.shape[1:], num_classes=Y_train.shape[1],
                            filters_block1=16, filters_block2=32, kernel_size=best_k,
                            activation="relu", kernel_initializer="he_normal", dense_units=100)
    final_res = train_and_evaluate(final_model, X_train, Y_train, X_test, Y_test,
                                   epochs=epochs, batch_size=batch_size,
                                   validation_split=validation_split,
                                   use_earlystopping=use_earlystopping,
                                   es_patience=es_patience,
                                   verbose=0)
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(final_res["model"].predict(X_test), axis=1)
    plot_confusion_matrix(y_true, y_pred, labels=[str(i) for i in range(Y_test.shape[1])],
                          title=f"CNN2 - Matriz confusión kernel {best_k[0]}x{best_k[1]}",
                          filename=f"CNN_tarea2_confmat_best_kernel_{best_k[0]}x{best_k[1]}.png")

    return results_sorted



def build_cnn_advanced(input_shape=(32,32,3),
                       num_classes=10,
                       blocks: List[int] = [32, 64],
                       convs_per_block: int = 2,
                       kernel_size: Tuple[int,int] = (3,3),
                       activation: str = 'relu',
                       kernel_initializer: str = 'he_normal',
                       weight_decay: float = 0.0,
                       use_batchnorm: bool = True,
                       dropout: float = 0.0,
                       global_pool: bool = True,
                       data_augmentation: bool = False) -> keras.models.Model:
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    inputs = layers.Input(shape=input_shape)
    x = inputs

    if data_augmentation:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomRotation(0.02),
        ], name="data_augmentation")
        x = aug(x)

    for filters in blocks:
        for i in range(convs_per_block):
            x = layers.Conv2D(filters, kernel_size, padding="same",
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=reg)(x)
            if use_batchnorm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
        x = layers.MaxPooling2D((2,2))(x)

    if global_pool:
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)

    if dropout and dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg)(x)
    if dropout and dropout > 0.0:
        x = layers.Dropout(min(0.5, dropout*1.0))(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def tareaCNN3(repetitions: int = 3,
              epochs: int = 80,
              batch_size: int = 128,
              validation_split: float = 0.1,
              use_earlystopping: bool = True,
              es_patience: int = 10,
              verbose: int = 0):
    X_train, Y_train, X_test, Y_test = load_preprocess_cnn()
    num_classes = Y_train.shape[1]
    input_shape = X_train.shape[1:]
    results = []

    class BuildParams(NamedTuple):
        blocks: List[int]
        convs_per_block: int = 2
        kernel_size: Tuple[int, int] = (3, 3)
        activation: str = 'relu'
        kernel_initializer: str = 'he_normal'
        weight_decay: float = 0.0
        dropout: float = 0.0
        batchnorm: bool = True
        global_pool: bool = True
        data_augmentation: bool = False

    class TrainParams(NamedTuple):
        epochs: int
        batch_size: int
        validation_split: float = 0.1
        earlystopping: bool = True
        es_patience: int = 10
        verbose: int = 0

    class ModelTemplate(NamedTuple):
        buildparams: BuildParams
        trainparams: TrainParams
        const_buildparams: Tuple = (input_shape, num_classes)


    candidates = [
        ModelTemplate(
            buildparams=BuildParams(
                blocks=[32, 64],
                weight_decay=1e-4,
                dropout=0.30,
                batchnorm=True,
                global_pool=True,
                data_augmentation=True
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),

        ModelTemplate(
            buildparams=BuildParams(
                blocks=[32, 64, 128],
                weight_decay=1e-4,
                dropout=0.40,
                batchnorm=True,
                global_pool=True,
                data_augmentation=True
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),

        ModelTemplate(
            buildparams=BuildParams(
                blocks=[64, 128],
                weight_decay=1e-4,
                dropout=0.45,
                batchnorm=False,
                global_pool=True,
                data_augmentation=True
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),

        ModelTemplate(
            buildparams=BuildParams(
                blocks=[32, 64],
                weight_decay=1e-5,
                dropout=0.20,
                batchnorm=True,
                global_pool=True,
                data_augmentation=False
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),

        ModelTemplate(
            buildparams=BuildParams(
                blocks=[32, 64],
                kernel_size=(5, 5),
                weight_decay=1e-4,
                dropout=0.30,
                batchnorm=True,
                global_pool=True,
                data_augmentation=True
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),

        ModelTemplate(
            buildparams=BuildParams(
                blocks=[32, 64, 128],
                weight_decay=1e-4,
                dropout=0.40,
                batchnorm=True,
                global_pool=False,
                data_augmentation=True
            ),
            trainparams=TrainParams(epochs=80, batch_size=128)
        ),
    ]

    for idx, c in enumerate(candidates):
        bp, tp = c.buildparams, c.trainparams

        label = (
            f"{'+'.join(map(str, bp.blocks))}"
            f"_k{bp.kernel_size[0]}"
            f"_do{bp.dropout}"
            f"_wd{bp.weight_decay}"
            f"{'_bn' if bp.batchnorm else ''}"
            f"{'_aug' if bp.data_augmentation else ''}"
            f"{'_gap' if bp.global_pool else '_flat'}"
        )

        print(f"\nProbando candidato {idx}: {label}")

        accs, times, raws = [], [], []

        for rep in range(repetitions):
            ensure_reproducibility(seed=1000 + rep)

            model = build_cnn_advanced(
                input_shape=input_shape,
                num_classes=num_classes,
                blocks=bp.blocks,
                convs_per_block=bp.convs_per_block,
                kernel_size=bp.kernel_size,
                activation=bp.activation,
                kernel_initializer=bp.kernel_initializer,
                weight_decay=bp.weight_decay,
                use_batchnorm=bp.batchnorm,
                dropout=bp.dropout,
                global_pool=bp.global_pool,
                data_augmentation=bp.data_augmentation
            )

            callbacks = []
            if tp.earlystopping:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=tp.es_patience,
                        restore_best_weights=True,
                        verbose=0
                    )
                )
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=0
                )
            )

            t0 = time.time()
            history = model.fit(
                X_train, Y_train,
                validation_split=tp.validation_split,
                epochs=tp.epochs,
                batch_size=tp.batch_size,
                callbacks=callbacks,
                verbose=tp.verbose
            )
            train_time = time.time() - t0

            test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

            accs.append(test_acc)
            times.append(train_time)
            raws.append({
                "model": model,
                "history": history.history,
                "test_acc": test_acc,
                "train_time": train_time
            })

            if rep == 0:
                plot_loss_acc(
                    history.history,
                    title=f"CNN3 - {label}",
                    filename=f"CNN3_{label}.png"
                )

            tf.keras.backend.clear_session()

        results.append({
            "label": label,
            "test_acc": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "train_time": float(np.mean(times)),
            "raws": raws
        })

        print(f"{label}: acc={np.mean(accs):.4f}, std_deviation: {np.std(accs):.4f}")

    results_sorted = sorted(results, key=lambda r: r["test_acc"], reverse=True)
    plot_bars_results(results_sorted, title="CNN3 - Comparativa", filename="CNN3_comparativa.png")

    best = results_sorted[0]
    best_model = best["raws"][0]["model"]

    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    plot_confusion_matrix(
        y_true, y_pred,
        labels=[str(i) for i in range(num_classes)],
        title=f"CNN3 - Confusión ({best['label']})",
        filename="CNN3_confusion_best.png"
    )

    return results_sorted


if __name__ == "__main__":
    #res = tareaMLP1()
    #res = tareaMLP2()
    #res = tareaMLP3()
    #res = tareaMLP4()
    #res = tareaMLP5()
    #res = tareaMLP6()
    #res = tareaMLP7()

    #res = tareaCNN1()
    #res = tareaCNN2()
    res = tareaCNN3()
