#!/usr/bin/env python3
"""Deep learning from scratch."""
import csv
import itertools
import json
import numpy as np
from typing import Tuple
from scipy.sparse import coo_matrix
import sys


def label_binarize(y: np.ndarray, classes: list) -> np.ndarray:
    """Convert the array of labels to an array of one hot vectors."""
    return np.hstack([y == c for c in classes])


def softmax(X: np.ndarray, axis: int) -> np.ndarray:
    """Normalize a vector of weights to probabilities."""
    e = np.exp(X)
    return e / e.sum(axis=axis)[np.newaxis].T


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multiclass cross entropy cost."""
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy."""
    return (y_true.argmax(axis=1) == y_pred.argmax(axis=1)).mean()


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    y_true, y_pred = y_true.argmax(axis=1), y_pred.argmax(axis=1)
    c = (np.ones_like(y_true), (y_true, y_pred))
    c = coo_matrix(c, shape=(CLASSES, CLASSES)).toarray()
    d = np.diag(c)
    precision, recall = d / c.sum(axis=1), d / c.sum(axis=0)
    return 2 * (precision * recall / (precision + recall)).mean()


def binary_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """One vs rest ROC AUC."""
    ix = np.argsort(y_pred)[::-1]
    y_true, y_pred = y_true[ix], y_pred[ix]
    distinct_value_indices = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = y_true.cumsum()[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_pred[threshold_idxs]
    c = np.logical_or(np.diff(fps, 2), np.diff(tps, 2))
    optimal_idxs = np.where(np.r_[True, c, True])[0]
    fps, tps = fps[optimal_idxs], tps[optimal_idxs]
    fps, tps = np.r_[0, fps], np.r_[0, tps]
    fpr, tpr = fps / fps[-1], tps / tps[-1]
    return np.trapz(tpr, fpr)


def roc_auc_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic Curve."""
    res = [binary_roc_auc(y_true[:, c], y_pred[:, c]) for c in classes]
    return np.mean(res)


class Dense:
    """A fully connected layer."""
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 activation: str = '',
                 l2: float = 0,
                 dropout: float = 0):
        """Randomly initialize weights."""
        bound = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(low=-bound, high=bound, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self._activation, self._l2, self._dropout = activation, l2, dropout

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        """Apply hidden layer and any additional transformations."""
        self._X = X
        ret = X @ self.W + self.b
        if self._activation == 'relu':
            ret[ret < 0] = 0
            self._next_X = ret
        if train:
            self._D = (np.random.rand(*ret.shape) >=
                       self._dropout) / (1 - self._dropout)
            ret *= self._D
        return ret

    def backward(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient for linear, activation and dropout."""
        delta *= self._D
        if self._activation == 'relu':
            delta *= self._next_X > 0
        grad_W = (self._X.T @ delta) / len(delta) + self._l2 * self.W
        return delta @ self.W.T, grad_W


class BatchNorm:
    """Batch normalization layer."""
    def __init__(self, n: int, momentum: float, epsilon: float):
        """Initialize weights."""
        self.W, self.b = np.ones(n), np.zeros(n)
        self._momentum, self._epsilon = momentum, epsilon
        self._running_mean, self._running_var = np.zeros(n), np.ones(n)

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        """Center and scale input data."""
        mean, var = self._running_mean, self._running_var
        if train:
            mean, var = np.mean(X, axis=0), np.var(X, axis=0)
            self._running_mean *= self._momentum
            self._running_mean += (1 - self._momentum) * mean
            self._running_var *= self._momentum
            self._running_var += (1 - self._momentum) * var
        self._std_inv = 1 / np.sqrt(var + self._epsilon)
        self._X = (X - mean) * self._std_inv
        return self.W * self._X + self.b

    def backward(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient for batch normalization."""
        dx = delta * self.W
        grad_W = (delta * self._X).mean(axis=0)
        return self._std_inv * (dx - dx.mean(axis=0) - self._X *
                                (dx * self._X).mean(axis=0)), grad_W


class Classifier:
    """Multiclass classifier implemented as a multi layer perceptron."""
    def __init__(self, layers: list):
        """Construct neural network."""
        self._layers = layers

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            batch_size: int,
            epochs: int,
            lr: float,
            momentum: float = 0) -> None:
        """Train the network with early stopping on training loss."""
        v_W = [np.zeros_like(layer.W) for layer in self._layers]
        v_b = [np.zeros_like(layer.b) for layer in self._layers]
        data, best = np.hstack((X, y)), np.Inf
        progress = {
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'momentum': momentum
        }
        for epoch in range(epochs):
            np.random.shuffle(data)
            loss = 0
            for batch in np.array_split(data, len(data) // batch_size):
                y_pred = self.predict_proba(batch[:, :-y.shape[1]], True)
                y_true = batch[:, -y.shape[1]:]
                cost, delta = log_loss(y_true, y_pred), y_pred - y_true
                for i, layer in reversed(list(enumerate(self._layers))):
                    grad_b = delta.mean(axis=0)
                    delta, grad_W = layer.backward(delta)
                    v_W[i] *= momentum
                    v_W[i] += lr * grad_W
                    v_b[i] *= momentum
                    v_b[i] += lr * grad_b
                    layer.W -= v_W[i]
                    layer.b -= v_b[i]
                loss += len(batch) * cost / len(data)
            progress['epoch'] = epoch + 1
            progress['logloss'] = loss
            for (t, X_val, y_val) in ('train', X, y), ('test', X_test, y_test):
                y_pred = self.predict_proba(X_val)
                for m in metrics:
                    progress[f'{t}_{m.__name__}'] = m(y_val, y_pred)
            print(json.dumps(progress), file=sys.stderr)
            if loss > best or np.isnan(loss):
                break
            best = loss

    def predict_proba(self, X: np.ndarray, train: bool = False) -> np.ndarray:
        """Run the network and apply softmax to get probabilities."""
        for layer in self._layers:
            X = layer(X, train)
        return softmax(X, axis=1)


[X, y, X_test, y_test] = [
    np.load(f'data/{k}.npy')
    for k in ('train_data', 'train_label', 'test_data', 'test_label')
]
CLASSES = 10
classes = list(range(CLASSES))
y = label_binarize(y, classes=classes)
y_test = label_binarize(y_test, classes=classes)
combinations = itertools.product([0, 0.2], [0, 0.01], [0, 0.9])
metrics = [log_loss, acc, f1_macro, roc_auc_macro]
fields = ['Dropout', 'L2', 'Momentum'] + [
    f'{t} {m.__name__}{s}' for t in ['Train', 'Test'] for m in metrics
    for s in ['', ' std']
]
w = csv.DictWriter(sys.stdout, fieldnames=fields)
w.writeheader()
trials = 5
train = np.zeros((trials, len(metrics)))
test = np.zeros((trials, len(metrics)))
for dropout, l2, momentum in combinations:
    row = {'Dropout': dropout, 'L2': l2, 'Momentum': momentum}
    for i in range(trials):
        model = Classifier([
            BatchNorm(X.shape[1], 0.99, 0.001),
            Dense(X.shape[1], 128, activation='relu', dropout=dropout, l2=l2),
            BatchNorm(128, 0.99, 0.001),
            Dense(128, 64, activation='relu', dropout=dropout, l2=l2),
            BatchNorm(64, 0.99, 0.001),
            Dense(64, y.shape[1]),
        ])
        model.fit(X, y, X_test, y_test, 1000, 50, 0.1, momentum=momentum)
        y_pred = model.predict_proba(X)
        y_test_pred = model.predict_proba(X_test)
        for k, m in enumerate(metrics):
            train[i][k] = m(y, y_pred)
            test[i][k] = m(y_test, y_test_pred)
    for t, a in ('Train', train), ('Test', test):
        mean, std = a.mean(axis=0), a.std(axis=0)
        for n, m, s in zip(metrics, mean, std):
            row[f'{t} {m.__name__}'] = m
            row[f'{t} {m.__name__} std'] = s
    w.writerow(row)
