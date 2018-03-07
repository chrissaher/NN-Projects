import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LinearRegression(object):
    """docstring for LinearRegression."""
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weights = None
        self.bias = None

    def random_mini_batches(self, X, Y, batch_size = 64):
        """Pending"""
        return [X], [Y]

    def fit(self, X, Y, epochs = 1000, batch_size = 64, lr = 0.0001, l2_reg = 0, print_rate = 50):
        tf.reset_default_graph()

        m, n = X.shape

        x = tf.placeholder(tf.float32, shape=(None, n), name='x')
        y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
        W = tf.Variable(np.random.randn(n, 1), dtype = 'float32', name='W')
        b = tf.Variable(np.random.randn(n), dtype = 'float32', name='b')
        z = tf.add(tf.matmul(x,W), b)

        # Regularization
        regularizers = tf.nn.l2_loss(W)

        # Mean square error
        cost = tf.reduce_sum(tf.pow(z - y, 2) / (2 * m) + l2_reg * regularizers)

        #Gradient descent
        gd = tf.train.GradientDescentOptimizer(learning_rate = lr, name = 'gd')
        optimizer = gd.minimize(cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                x_shuffle, y_shuffle = self.random_mini_batches(X, Y, batch_size)

                for (x_train, y_train) in zip(x_shuffle, y_shuffle):
                    sess.run(optimizer, feed_dict={'x:0':x_train, 'y:0':y_train})

                if epoch % print_rate == 0:
                    current_cost = sess.run(cost, feed_dict={'x:0':X, 'y:0':Y})
                    print('epoch: %3d  cost: %11.9lf'%(epoch, current_cost))
                    # print('W: ', sess.run(W))
                    # print('b: ', sess.run(b))
                    # print('-' * 5)

            self.weights = sess.run(W)
            self.bias = sess.run(b)
            training_cost = sess.run(cost, feed_dict={'x:0':X, 'y:0':Y})
            print('Training cost: ', training_cost)


    def predict(self, X):
        W, b = self.weights, self.bias
        return np.dot(X, W) + b

    def get_weights_bias(self):
        return np.array(self.weights), self.bias



if __name__ == '__main__':
    # ax^2 + bx + c = 0
    a = 4
    b = 6
    c = 2

    m = 20
    n = 3

    epochs = 15000
    lr = 0.1
    print_rate=5000
    l2_reg = 0

    X = np.ones((m, n))
    Y = np.ones((m, 1))
    for j in range(m):
        x = np.random.rand()
        y = a * x * x + b * x + c
        for i in range(n - 1):
            X[j][i + 1] = X[j][i] * x
        Y[j][0] = y

    linreg = LinearRegression()
    print('epochs: %d learning_rate: %.04lf'%(epochs, lr))
    print('-' * 30)

    print('Training with regularizacion: ',l2_reg)
    linreg.fit(X, Y, epochs=epochs, lr =lr, print_rate=print_rate)
    coef, bias = linreg.get_weights_bias()
    print('Expected coefficients: A: %.04lf  B: %.04lf  C: %.04lf'%(a, b, c))
    print('Predicted coefficients: A: %.04lf  B: %.04lf  C: %.04lf'%(coef[2], coef[1], coef[0]))
    print('Bias :  A: %.04lf  B: %.04lf  C: %.04lf'%(bias[2], bias[1], bias[0]))
    print('-' * 30)

    l2_reg = 0.001
    print('Training with regularizacion: ',l2_reg)
    linreg.fit(X, Y, epochs=epochs, lr=lr, l2_reg=l2_reg, print_rate =print_rate)
    coef, bias = linreg.get_weights_bias()
    print('Expected coefficients: A: %.04lf  B: %.04lf  C: %.04lf'%(a, b, c))
    print('Predicted coefficients: A: %.04lf  B: %.04lf  C: %.04lf'%(coef[2], coef[1], coef[0]))
    print('Bias :  A: %.04lf  B: %.04lf  C: %.04lf'%(bias[2], bias[1], bias[0]))
    print('-' * 30)
