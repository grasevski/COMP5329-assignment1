#!/usr/bin/env python3
"""Deep learning from scratch."""
import numpy as np
from typing import List


def label_binarize(y: np.ndarray, classes: List[int]) -> np.ndarray:
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
                 dropout: float = 0):
        """Randomly initialize weights."""
        bound = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(low=-bound, high=bound, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self._activation = activation
        self._dropout = dropout

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        """Apply hidden layer and any additional transformations."""
        self._X = X
        ret = X @ self.W + self.b
        if self._activation == 'relu':
            ret[ret < 0] = 0
        self._D = (np.random.rand(*ret.shape) >=
                   self._dropout) / (1 - self._dropout)
        ret *= self._D
        return ret

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Calculate gradient for linear, activation and dropout."""
        self.grad_W = self._X.T @ delta
        self.grad_b = delta.sum(axis=0)
        if self._activation == 'relu':
            delta[delta < 0] = 0
        delta *= self._D
        return delta @ self.W.T


class BatchNorm:
    """Batch normalization layer."""
    def __init__(self, n: int, momentum: float, epsilon: float):
        """Initialize weights."""
        self.W = np.ones(n)
        self.b = np.zeros(n)
        self._momentum = momentum
        self._epsilon = epsilon
        self._running_mean = np.zeros(n)
        self._running_var = np.ones(n)

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        """Center and scale input data."""
        mean = self._running_mean
        var = self._running_var
        if train:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self._running_mean *= self._momentum
            self._running_mean += (1 - self._momentum) * mean
            self._running_var *= self._momentum
            self._running_var += (1 - self._momentum) * var
        self._std_inv = 1 / np.sqrt(var + self._epsilon)
        self._X = (X - mean) * self._std_inv
        return self.W * self._X + self.b

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Calculate gradient for batch normalization."""
        self.grad_W = (delta * self._X).sum(axis=0)
        self.grad_b = delta.sum(axis=0)
        dx = delta * self.W
        return self._std_inv * (dx - dx.mean(axis=0) - self._X *
                                (dx * self._X).mean(axis=0))


class Classifier:
    """Multiclass classifier implemented as a multi layer perceptron."""
    def __init__(self, layers: list):
        """Construct neural network."""
        self.layers = layers

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            epochs: int,
            lr: float,
            momentum: float = 0,
            weight_decay: float = 0) -> None:
        """Train the network with early stopping on training loss."""
        v_W = [np.zeros_like(layer.W) for layer in self.layers]
        v_b = [np.zeros_like(layer.b) for layer in self.layers]
        data, best = np.hstack((X, y)), np.Inf
        for _ in range(epochs):
            np.random.shuffle(data)
            loss = 0
            for batch in np.array_split(data, len(data) // batch_size):
                y_pred = self.predict_proba(batch[:, :-y.shape[1]], True)
                y_true = batch[:, -y.shape[1]:]
                cost = log_loss(y_true, y_pred)
                delta = y_pred - y_true
                for layer in reversed(self.layers):
                    delta = layer.backward(delta)
                for i, layer in enumerate(self.layers):
                    v_W[i] = momentum * v_W[i] + lr * (
                        layer.grad_W + 2 * weight_decay * layer.W)
                    v_b[i] = momentum * v_b[i] + lr * (
                        layer.grad_b + 2 * weight_decay * layer.b)
                    layer.W -= v_W[i]
                    layer.b -= v_b[i]
                loss += len(batch) * cost / len(data)
            print(loss)
            if loss > best:
                break
            best = loss

    def predict_proba(self, X: np.ndarray, train: bool = False) -> np.ndarray:
        """Run the network and apply softmax to get probabilities."""
        for layer in self.layers:
            X = layer(X, train)
        return softmax(X, axis=1)


[X, y, X_test, y_test] = [
    np.load(f'data/{k}.npy')
    for k in ('train_data', 'train_label', 'test_data', 'test_label')
]
classes = list(range(10))
y = label_binarize(y, classes=classes)
y_test = label_binarize(y_test, classes=classes)
model = Classifier([
    Dense(X.shape[1], 1024, activation='relu', dropout=0.5),
    BatchNorm(1024, 0.99, 0.001),
    Dense(1024, 512, activation='relu', dropout=0.5),
    BatchNorm(512, 0.99, 0.001),
    Dense(512, y.shape[1]),
])
model.fit(X, y, 1000, 100, 1e-5, momentum=0.9, weight_decay=0.01)
y_pred = model.predict_proba(X)
loss = log_loss(y, y_pred)
print(f'train: {loss}')
y_pred = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred)
print(f'test: {loss}')
