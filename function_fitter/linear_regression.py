import tensorflow as tf
import numpy as np

class LinearRegression(object):
    """docstring for LinearRegression."""
    def __init__(self):
        super(LinearRegression, self).__init__()

    def random_mini_batches(self, X, Y, batch_size = 64):
        """Pending"""
        return X, Y

    def initialize_parameter(self, X):
        m = X.shape[0]
        nx = np.ones((m,3))
        for j in range(m):
            for i in range(2):
                nx[j][i + 1] = nx[j][i] * X[j]
        return nx

    def fit(self, X, Y, epochs = 1000, batch_size = 64, lr = 0.0001):
        X = self.initialize_parameter(X)
        m, n = X.shape

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        W = tf.Variable(np.random.randn(m, n), dtype = 'float32', name='W')
        b = tf.Variable(np.random.randn(n), dtype = 'float32', name='b')
        z = tf.add(tf.multiply(x,W), b)

        # Mean square error
        cost = tf.reduce_sum(tf.pow(z - y, 2)) / (2 * m)

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

                if epoch % 50 == 0:
                    current_cost = sess.run(cost, feed_dict={'x:0':x_shuffle, 'y:0':y_shuffle})
                    print('epoch: %3d  cost: %11.9lf b: %11.9lf'%(epoch, current_cost, sess.run(b)))
                    print('W: ', sess.run(W))
                    # print('epoch: %3d  cost: %11.9lf  W: %11.9lf  b: %11.9lf'%(epoch, current_cost, sess.run(W), sess.run(b)))


            training_cost = sess.run(cost, feed_dict={'x:0':X, 'y:0':Y})
            print('Training cost: ', training_cost)





if __name__ == '__main__':
    # ax^2 + bx + c = 0
    a = 4
    b = 6
    c = 2

    X = []
    Y = []
    m = 20
    for i in range(m):
        x = np.random.rand()
        y = a * x * x + b * x + c
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    # X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    # Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221, 2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    linreg = LinearRegression()
    linreg.fit(X, Y, epochs = 1000, lr = 0.01)
    print('FINISH')
