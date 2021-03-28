#!/usr/bin/env python3
"""Deep learning from scratch."""
import json
import numpy as np
import optuna
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
        self._activation, self._dropout = activation, dropout

    def __call__(self, X: np.ndarray, train: bool) -> np.ndarray:
        """Apply hidden layer and any additional transformations."""
        self._X = X
        ret = X @ self.W + self.b
        if self._activation == 'relu':
            ret[ret < 0] = 0
        if train:
            self._D = (np.random.rand(*ret.shape) >=
                       self._dropout) / (1 - self._dropout)
            ret *= self._D
        return ret

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Calculate gradient for linear, activation and dropout."""
        self.grad_W, self.grad_b = self._X.T @ delta, delta.sum(axis=0)
        delta *= self._D
        delta = delta @ self.W.T
        if self._activation == 'relu' or True:
            delta *= self._X > 0
        return delta


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
        self.grad_W = (delta * self._X).sum(axis=0)
        self.grad_b = delta.sum(axis=0)
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
            momentum: float = 0,
            weight_decay: float = 0) -> None:
        """Train the network with early stopping on training loss."""
        v_W = [np.zeros_like(layer.W) for layer in self._layers]
        v_b = [np.zeros_like(layer.b) for layer in self._layers]
        data, best = np.hstack((X, y)), np.Inf
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
                    v_W[i] -= lr * (layer.grad_W + weight_decay * layer.W)
                    v_b[i] *= momentum
                    v_b[i] -= lr * (layer.grad_b + weight_decay * layer.b)
                    layer.W += v_W[i]
                    layer.b += v_b[i]
                loss += len(batch) * cost / len(data)
            print(f'epoch {epoch + 1}/{epochs} log loss: {loss}')
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


def objective(trial) -> float:
    n1 = trial.suggest_int('n1', y.shape[1], X.shape[1], log=True)
    n2 = trial.suggest_int('n2', y.shape[1], X.shape[1], log=True)
    model = Classifier([
        BatchNorm(X.shape[1], trial.suggest_uniform('m1', 0, 1), trial.suggest_loguniform('e1', 1e-9, 1)),
        Dense(X.shape[1], n1, activation='relu', dropout=trial.suggest_loguniform('d1', 1e-9, 1)),
        BatchNorm(n1, trial.suggest_uniform('m2', 0, 1), trial.suggest_loguniform('e2', 1e-9, 1)),
        Dense(n1, n2, activation='relu', dropout=trial.suggest_loguniform('d2', 1e-9, 1)),
        BatchNorm(n2, trial.suggest_uniform('m3', 0, 1), trial.suggest_loguniform('e3', 1e-9, 1)),
        Dense(n2, y.shape[1]),
    ])
    model.fit(X, y, trial.suggest_int('b', 1, len(X), log=True), 1000, trial.suggest_loguniform('lr', 1e-9, 1e-4), momentum=trial.suggest_uniform('momentum', 0, 1), weight_decay=trial.suggest_loguniform('weight_decay', 1e-9, 1))
    return log_loss(y_test, model.predict_proba(X_test))


def make(params: dict) -> list:
    return [
        BatchNorm(X.shape[1], params['m1'], params['e1']),
        Dense(X.shape[1], params['n1'], activation='relu', dropout=params['d1']),
        BatchNorm(params['n1'], params['m2'], params['e2']),
        Dense(params['n1'], params['n2'], activation='relu', dropout=params['d2']),
        BatchNorm(params['n2'], params['m3'], params['e3']),
        Dense(params['n2'], y.shape[1]),
    ]




#{"value": 2.0623634547747502, "params": {"n1": 126, "n2": 82, "m1": 0.31574289094103947, "e1": 4.633534155201676e-05, "d1": 0.00047475009162580007, "m2": 0.5800161698683498, "e2": 0.0001394505994988068, "d2": 0.00035707295916165716, "m3": 0.3087861681071012, "e3": 4.681352930833947e-08, "lr": 3.8083086753387537e-06, "momentum": 0.5466724735934154, "weight_decay": 5.430304659597914e-06}}
#{"n1": 66, "n2": 60, "m1": 0.9118555721913466, "e1": 2.4314972323867697e-06, "d1": 2.3876714806756635e-05, "m2": 0.044173149016412144, "e2": 0.9405022889233455, "d2": 4.989790137474286e-09, "m3": 0.49352730366205383, "e3": 0.5526208592051385, "lr": 2.8852447317380467e-05, "momentum": 0.45814875359108037, "weight_decay": 1.3468924099172088e-09}


#study = optuna.create_study()
#study.optimize(objective, n_trials=10000)
#trial = study.best_trial
#print(json.dumps({'value': trial.value, 'params': trial.params}))

model = Classifier([
    #Dense(X.shape[1], 1024, activation='relu', dropout=0.5),
    #BatchNorm(1024, 0.99, 0.001),
    #Dense(1024, 512, activation='relu', dropout=0.5),
    #BatchNorm(512, 0.99, 0.001),
    Dense(X.shape[1], y.shape[1], activation='relu'),
    Dense(y.shape[1], y.shape[1]),
])
model.fit(X, y, 100, 1000, 1e-5, momentum=0.9)
print(f'train log loss: {log_loss(y, model.predict_proba(X))}')
print(f'test log loss: {log_loss(y_test, model.predict_proba(X_test))}')
