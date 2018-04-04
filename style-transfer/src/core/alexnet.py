import numpy as np
import tensorflow as tf
from helpers import conv, fc, max_pool, lrn, dropout

class AlexNet(object):
    
    def __init__(self, x, keep_prob, num_classes, weights_path = 'DEFAULT'):
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - num_classes: int, number of classes of the new dataset
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights,
                        (if bvlc_alexnet.npy is not in the same folder)
        """
        # Parse input arguments
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        sz = int(np.prod(pool5.get_shape()[1:]))
        flattened = tf.reshape(pool5, [-1, sz])
        fc6 = fc(flattened, sz, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
        
        # For future model
        self.input = self.X
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.fc6 = fc6
        self.fc7 = fc7
        self.fc8 = fc8

    def load_initial_weights(self, session):
        
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            with tf.variable_scope(op_name, reuse = True):

                # Loop over list of weights/biases and assign them to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable = False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable = False)
                        session.run(var.assign(data))
                        
    def export(self):
        model = dict()
        model['input'] = self.X
        model['conv1'] = self.conv1
        model['conv2'] = self.conv2
        model['conv3'] = self.conv3
        model['conv4'] = self.conv4
        model['conv5'] = self.conv5
        model['fc6'] = self.fc6
        model['fc7'] = self.fc7
        model['fc8'] = self.fc8
        return model