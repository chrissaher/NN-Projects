from rnn_cell import RNN_Cell
from helpers import smooth, get_initial_loss, print_sample, softmax
import numpy as np

class RNN(object):
    """docstring for RNN."""
    def __init__(self, vocab_size=None, num_cells=None):
        super(RNN, self).__init__()
        self.n_x = vocab_size
        self.n_y = vocab_size
        self.n_a = num_cells
        self.cell = RNN_Cell()

    def initialize_parameters(self):
        n_x, n_y, n_a = self.n_x, self.n_y, self.n_a
        np.random.seed(1)
        Wx = np.random.randn(n_a, n_x) * 0.01 # input to hidden
        Wa = np.random.randn(n_a, n_a) * 0.01 # hidden to hidden
        Wy = np.random.randn(n_y, n_a) * 0.01 # hidden tp output
        b = np.zeros((n_x, 1)) * 0.01 # hidden bias
        by = np.zeros((n_y, 1)) * 0.01 # output bias
        parameters = {'Wx':Wx, 'Wa':Wa, 'Wy':Wy, 'b':b, 'by':by}
        return parameters

    def forward(self, X, a0, parameters):
        x, a, y_pred = {}, {}, {}
        vocab_size, cell = self.n_x, self.cell

        a[-1] = np.copy(a0)
        for t in range(len(X)):
            x[t] = np.zeros((vocab_size, 1))
            if X[t] != None:
                x[t][X[t]] = 1

            a[t], y_pred[t] = cell.forward(x[t], a[t - 1], parameters)

        cache = (y_pred, a, x)
        return cache

    def calculate_lost(self, Y, cache):
        loss = 0
        (y_pred, a , x) = cache
        for t in range(len(Y)):
            loss -= np.log(y_pred[t][Y[t], 0])
        return loss

    def backpropagation(self, X, Y, parameters, cache):
        (y_pred, a, x) = cache
        cell = self.cell

        gradients = {}
        gradients['dWx'] = np.zeros_like(parameters['Wx'])
        gradients['dWa'] = np.zeros_like(parameters['Wa'])
        gradients['dWy'] = np.zeros_like(parameters['Wy'])
        gradients['db'] = np.zeros_like(parameters['b'])
        gradients['dby'] = np.zeros_like(parameters['by'])
        gradients['da_next'] = np.zeros_like(a[0])

        for t in reversed(range(len(X))):
            dy = np.copy(y_pred[t])
            dy[Y[t]] -= 1
            gradients = cell.backward(dy, x[t], a[t], a[t - 1], parameters, gradients)
        return gradients, a

    def clip(self, gradients, maxValue = 10):
        dWx = gradients['dWx']
        dWa = gradients['dWa']
        dWy = gradients['dWy']
        db = gradients['db']
        dby = gradients['dby']

        for gradient in [dWx, dWa, dWy, db, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)

        gradients = {'dWx':dWx, 'dWa':dWa, 'dWy':dWy, 'db':db, 'dby':dby}
        return gradients

    def update_parameters(self, lr, parameters, gradients):
        return self.cell.update(lr, parameters, gradients)

    def sample(self, parameters, char_to_ix, sample_limit = 50):
        vocab_size, n_a = self.n_x, self.n_a

        Wx = parameters['Wx']
        Wa = parameters['Wa']
        Wy = parameters['Wy']
        b = parameters['b']
        by = parameters['by']

        x = np.zeros((vocab_size, 1))
        a_prev = np.zeros((n_a, 1))
        idx = -1
        indices = []
        counter = 0
        newline_idx = char_to_ix['\n']

        while idx != newline_idx and counter != sample_limit:
            a = np.tanh(np.dot(Wx, x) + np.dot(Wa, a_prev) + b)
            y = softmax(np.dot(Wy, a) + by)

            idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
            indices.append(idx)

            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            a_prev = a

            counter += 1

        if counter == 50:
            indices.append(newline_idx)
        return indices

    def fit(self, X, num_iter = 100, lr=1e-5, char_to_ix=None, dino_names=10):
        parameters = self.initialize_parameters()
        n_x, n_y, n_a = self.n_x, self.n_y, self.n_a

        loss = get_initial_loss(vocab_size, dino_names)

        a_prev = np.zeros((n_a, 1))
        for j in range(num_iter):
            example = X[j % len(examples)]
            x = [None] + [char_to_ix[ch] for ch in example]
            y = x[1:] + [char_to_ix['\n']]

            cache = self.forward(x, a_prev, parameters)
            curr_loss = self.calculate_lost(y, cache)
            gradients, a = self.backpropagation(x, y, parameters, cache)
            gradients = self.clip(gradients, maxValue=5)
            parameters = self.update_parameters(lr, parameters, gradients)

            loss = smooth(loss, curr_loss)

            if j % 2000 == 0:
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
                for name in range(dino_names):
                    sampled_indices = self.sample(parameters, char_to_ix)
                    print_sample(sampled_indices, ix_to_char)
                print('\n')

        return parameters

if __name__ == '__main__':
    data = open('dinos.txt', 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

    rnn = RNN(vocab_size = vocab_size, num_cells = 27)
    with open('dinos.txt') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    print('*' * 20)
    rnn.fit(examples, num_iter=40000, lr=0.01, char_to_ix=char_to_ix)
    print('*' * 20)
