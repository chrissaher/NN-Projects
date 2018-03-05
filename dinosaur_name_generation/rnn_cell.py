import numpy as np
from helpers import softmax

class RNN_Cell(object):
    """docstring for RNN_Cell."""

    def __init__(self):
        super(RNN_Cell, self).__init__()

    def forward(self, xt, a_prev, parameters):
        Wx = parameters['Wx']
        Wa = parameters['Wa']
        Wy = parameters['Wy']
        b = parameters['b']
        by = parameters['by']

        a_next = np.tanh(np.dot(Wx, xt) + np.dot(Wa, a_prev) + b)
        yt = softmax(np.dot(Wy, a_next) + by)
        cache = {'a_prev':a_prev, 'a_next':a_next, 'xt':xt, 'yt':yt}
        return a_next, yt

    def backward(self, dy, xt, a_next, a_prev, parameters, gradients):
        Wa, Wy = parameters['Wa'], parameters['Wy']

        gradients['dWy'] += np.dot(dy, a_next.T)
        gradients['dby'] += dy
        da = np.dot(Wy.T, dy) + gradients['da_next']
        daraw = (1 - a_next * a_next) * da # backprop of tanh
        gradients['db'] += daraw
        gradients['dWx'] += np.dot(daraw, xt.T)
        gradients['dWa'] += np.dot(daraw, a_prev.T)
        gradients['da_next'] = np.dot(Wa.T, daraw)
        return gradients

    def update(self, lr, parameters, gradients):
        parameters['Wx'] += -lr * gradients['dWx']
        parameters['Wa'] += -lr * gradients['dWa']
        parameters['Wy'] += -lr * gradients['dWy']
        parameters['b'] += -lr * gradients['db']
        parameters['by'] += -lr * gradients['dby']
        return parameters
