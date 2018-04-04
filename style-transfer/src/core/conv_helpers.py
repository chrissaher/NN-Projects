import numpy as np
import tensorflow as tf

data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="bytes").item()

def weights(layer_name):
    layer = data[layer_name]
    W = layer[0]
    b = layer[1]
    return W, b

def conv2d(prev_layer, layer_name, strides=[1,1,1,1], groups = 1, relu = True):
    W, b = weights(layer_name)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    
    convolve = lambda prev, w: tf.nn.conv2d(prev, w, strides, padding='SAME')
    
    if groups == 1:
        conv = convolve(prev_layer, W)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=prev_layer)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
        output_groups = [convolve(prev, w) for (prev, w) in zip(input_groups, weight_groups)]
        
        conv = tf.concat(axis=3, values=output_groups)
    
    
    conv_bias = tf.nn.bias_add(conv, b)
    if relu == True:
        return tf.nn.relu(conv_bias)
    return conv

def maxpool(prev_layer):
    return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def export_graph(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL):
    graph = dict()
    graph['input'] = tf.Variable(np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)), dtype=tf.float32)

    graph['conv1'] = conv2d(graph['input'], 'conv1', strides=[1,4,4,1])
    graph['pool1'] = maxpool(graph['conv1'])
    
    graph['conv2'] = conv2d(graph['pool1'], 'conv2', groups=2)
    graph['pool2'] = maxpool(graph['conv2'])
    
    graph['conv3'] = conv2d(graph['pool2'], 'conv3')
    graph['pool3'] = maxpool(graph['conv3'])
    
    graph['conv4'] = conv2d(graph['pool3'], 'conv4', groups=2)
    graph['pool4'] = maxpool(graph['conv4'])
    
    graph['conv5'] = conv2d(graph['pool4'], 'conv5', groups=2)
    return graph