#!/usr/bin/env python3
"""Deep learning from scratch."""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from typing import List


class Layer:
    """A fully connected layer."""
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 activation: str = '',
                 dropout: float = 0):
        bound = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(low=-bound, high=bound, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self._activation = activation
        self._dropout = dropout

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        self._X = X
        ret = X @ self.W + self.b
        if self._activation == 'relu':
            ret[ret < 0] = 0
        self._D = (np.random.rand(*ret.shape) >=
                   self._dropout) / (1 - self._dropout)
        ret *= self._D
        return ret

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Update the gradients."""
        self.grad_W = self._X.T @ delta
        self.grad_b = np.sum(delta, axis=0)
        if self._activation == 'relu':
            delta[delta < 0] = 0
        delta *= self._D
        return delta @ self.W.T


class Classifier:
    """Multiclass classifier implemented as a multi layer perceptron."""
    def __init__(self, layers: List[Layer]):
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
    Layer(X.shape[1], 181, activation='relu', dropout=0.1),
    Layer(181, y.shape[1]),
])
model.fit(X, y, 1000, 100, 1e-5, momentum=0.9, weight_decay=0.1)
y_pred = model.predict_proba(X)
loss = log_loss(y, y_pred)
print(f'train: {loss}')
y_pred = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred)
print(f'test: {loss}')
