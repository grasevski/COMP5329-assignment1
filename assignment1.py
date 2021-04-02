#!/usr/bin/env python3
"""Deep learning from scratch."""
import csv
import itertools
import json
import numpy as np
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

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Calculate gradient for linear, activation and dropout."""
        delta *= self._D
        if self._activation == 'relu':
            delta *= self._next_X > 0
        self.grad_W = (self._X.T @ delta) / len(delta) + self._l2 * self.W
        self.grad_b = delta.mean(axis=0)
        return delta @ self.W.T


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

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Calculate gradient for batch normalization."""
        self.grad_W = (delta * self._X).mean(axis=0)
        self.grad_b = delta.mean(axis=0)
        dx = delta * self.W
        return self._std_inv * (dx - dx.mean(axis=0) - self._X *
                                (dx * self._X).mean(axis=0))


class Classifier:
    """Multiclass classifier implemented as a multi layer perceptron."""
    def __init__(self, layers: list):
        """Construct neural network."""
        self._layers = layers

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
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
                    delta = layer.backward(delta)
                    v_W[i] *= momentum
                    v_W[i] += lr * layer.grad_W
                    v_b[i] *= momentum
                    v_b[i] += lr * layer.grad_b
                    layer.W -= v_W[i]
                    layer.b -= v_b[i]
                loss += len(batch) * cost / len(data)
            progress['epoch'] = epoch + 1
            progress['logloss'] = loss
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
classes = list(range(10))
y = label_binarize(y, classes=classes)
y_test = label_binarize(y_test, classes=classes)
combinations = itertools.product([0, 0.2], [0, 0.01], [0, 0.9])
fields = [
    'Dropout', 'L2', 'Momentum', 'Train', 'Train std', 'Test', 'Test std'
]
w = csv.DictWriter(sys.stdout, fieldnames=fields)
w.writeheader()
trials = 5
train, test = np.zeros(trials), np.zeros(trials)
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
        model.fit(X, y, 1000, 50, 0.1, momentum=momentum)
        y_pred = model.predict_proba(X)
        train[i] = log_loss(y, y_pred)
        y_pred = model.predict_proba(X_test)
        test[i] = log_loss(y_test, y_pred)
    row['Train'] = train.mean()
    row['Train std'] = train.std()
    row['Test'] = test.mean()
    row['Test std'] = test.std()
    w.writerow(row)
