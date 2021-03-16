#!/usr/bin/env python3
"""Neural networks from scratch."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize


# create a activation class
# for each time, we can initiale a activation function object with
# one specific function
# for example: f = Activation("tanh")  means we create a tanh
# activation function.
# you can define more activation functions by yourself, such as relu!
class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __init__(self, activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = lambda x: x if x > 0 else 0
            self.f_deriv = lambda a: int(a > 0)


# now we define the hidden layer for the mlp
# for example, h1 = HiddenLayer(10, 5, activation="tanh") means we
# create a layer with 10 dimension input and 5 dimension output,
# and using tanh activation function.
# notes: make sure the input size of hiddle layer should be matched
# with the output size of the previous layer!
class HiddenLayer(object):
    def __init__(self,
                 n_in,
                 n_out,
                 activation_last_layer='tanh',
                 activation='tanh',
                 W=None,
                 b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = None
        self.activation = Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the
        # initiallization
        self.W = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                   high=np.sqrt(6. / (n_in + n_out)),
                                   size=(n_in, n_out))
        if activation == 'logistic':
            self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out, )

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # the forward and backward progress for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None else
                       self.activation(lin_output))
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0)
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta


class MLP:
    """
    """

    # for initiallization, the code will create all layers
    # automatically based on the provided parameters.
    def __init__(self, layers, activation=[None, 'tanh', 'tanh']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic", "tanh" or "relu"
        """
        # initialize layers
        self.layers = []
        self.params = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(
                HiddenLayer(layers[i], layers[i + 1], activation[i],
                            activation[i + 1]))

    # forward progress: pass the information through the layers and
    # out the results of final output layer
    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    # define the objection/loss function, we use mean sqaure error
    # (MSE) as the loss
    # you can try other loss, such as cross entropy.
    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv
        error = y - y_hat
        loss = np.average(np.average(error ** 2, axis=0))
        delta = -error * activation_deriv(y_hat)
        return loss, delta

    def criterion_crossentropy(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv
        error = y - y_hat
        loss = log_loss(y, y_hat)
        delta = -error * activation_deriv(y_hat)
        return loss, delta

    # backward progress
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!
    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self, X, y, learning_rate=0.1, epochs=100, batch=1024):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of iterations over the training data
        :param batch: mini batch size
        """
        data, to_return = np.hstack((X, y)), np.zeros(epochs)
        for k in range(epochs):
            np.random.shuffle(data)
            for b in np.array_split(data, len(data) // batch):
                y_hat = self.forward(b[:, :-y.shape[1]])
                loss, delta = self.criterion_crossentropy(b[:, -y.shape[1]:], y_hat)
                self.backward(delta)
                self.update(learning_rate)
                to_return[k] += len(b) * loss / len(data)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new
    # data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros((x.shape[0], len(self.layers[-1].b)))
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output


[X, y, X_test, y_test] = [
    np.load(f'data/{k}.npy')
    for k in ('train_data', 'train_label', 'test_data', 'test_label')
]

lr = LogisticRegression(multi_class='ovr').fit(X, y.ravel())
y = label_binarize(y, classes=lr.classes_)
y_test = label_binarize(y_test, classes=lr.classes_)
y_pred = lr.predict_proba(X)
loss = log_loss(y, y_pred)
print(f'baseline train: {loss}')
y_pred = lr.predict_proba(X_test)
loss = log_loss(y_test, y_pred)
print(f'baseline: {loss}')

nn = MLP([X.shape[1], y.shape[1]], [None, 'logistic'])
loss = nn.fit(X, y, learning_rate=0.001)
print(f'nn train: {loss[-1]}')
y_pred = nn.predict(X_test)
loss = log_loss(y_test, y_pred)
print(f'nn: {loss}')
