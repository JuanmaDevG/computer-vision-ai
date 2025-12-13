import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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


def tareaMLP3(batch_sizes: List[int] = [16, 32, 64], repetitions: int = 3, use_earlystopping: bool = False):
    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for b in batch_sizes:
        label = f"bs_{b}"
        accs = []
        times = []
        histories = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=123 + rep)
            model = build_mlp(input_dim=X_train.shape[1], hidden_layers=[48], activation="relu", kernel_initializer="he_normal")
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=20, batch_size=b, validation_split=0.1,
                                     use_earlystopping=use_earlystopping, es_patience=5, verbose=0)
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


def tareaMLP5(neuron_counts: List[int] = [16, 32, 48, 64, 96], repetitions: int = 3, use_earlystopping: bool = False):
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
        configs = [[96], [48, 48], [32, 64], [64, 32], [32, 32, 32], [16, 32, 48]]
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


#TODO: investigate and make handmade
def tareaMLP7(repetitions: int = 3, use_earlystopping: bool = True):
    candidates = [
        {'hidden': [128, 64], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.3, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [256, 128], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.4, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [96, 48, 32], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.2, 'l2': 1e-4, 'batchnorm': True},
        {'hidden': [512], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.5, 'l2': 1e-3, 'batchnorm': False},
        {'hidden': [384, 256], 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.4, 'l2': 1e-4, 'batchnorm': True},
    ]
    candidates = [c for c in candidates if sum(c['hidden']) <= 1000 and len(c['hidden']) <= 6]

    X_train, Y_train, X_test, Y_test = load_preprocess_mlp()
    results = []
    for c in candidates:
        label = f"{'+'.join(map(str, c['hidden']))}_act-{c['activation']}_do-{c['dropout']}_l2-{c['l2']}"
        print(f"Probando candidate: {label}")
        accs = []
        times = []
        raws = []
        for rep in range(repetitions):
            ensure_reproducibility(seed=999 + rep)
            model = build_mlp(input_dim=X_train.shape[1],
                              hidden_layers=c['hidden'],
                              activation=c['activation'],
                              kernel_initializer=c['init'],
                              l2_reg=c['l2'],
                              dropout=c['dropout'],
                              use_batchnorm=c['batchnorm'])
            res = train_and_evaluate(model, X_train, Y_train, X_test, Y_test,
                                     epochs=50, batch_size=64, validation_split=0.1,
                                     use_earlystopping=use_earlystopping, es_patience=6, verbose=0)
            accs.append(res["test_acc"])
            times.append(res["train_time"])
            raws.append(res)
        results.append({"label": label, "test_acc": float(np.mean(accs)), "train_time": float(np.mean(times)), "raws": raws})
        print(f"{label}: mean_acc={np.mean(accs):.4f}, mean_time={np.mean(times):.1f}s")

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
    print("\nTop candidates (label, acc, time):")
    for r in results_sorted[:5]:
        print(r["label"], f"{r['test_acc']:.4f}", f"{r['train_time']:.1f}s")
    return results_sorted


# ---------------------------
# Plantillas CNN (por completar)
# ---------------------------
def build_cnn_template(input_shape=(32, 32, 3), num_classes=10):
    """
    Plantilla CNN sencilla (para implementar CNN1, CNN2, CNN3).
    Se deja como plantilla para que completes en la segunda parte.
    """
    model = keras.models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------
# Main: ejecuta la tarea deseada (descomenta la que quieras correr)
# ---------------------------
if __name__ == "__main__":
    # WARNING: dejar comentadas las tareas que no se usan

    # mlp1 = tareaMLP1()
    res2 = tareaMLP2()
    # res3 = tareaMLP3()
    # res4 = tareaMLP4()
    # res5 = tareaMLP5()
    # res6 = tareaMLP6()
    # res7 = tareaMLP7()

    #TODO: Aquí actividades de CNN
