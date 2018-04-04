from __future__ import print_function
from conv_helpers import *
from image_helpers import *
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf


class StyleTransfer(object):
    
    def __init__(self, content_image, style_image, fixed_width=360, fixed_height=360, fixed_channel=3):
        self.IMG_WIDTH = fixed_width
        self.IMG_HEIGHT = fixed_height
        self.IMG_CHANNEL = fixed_channel

        #img_content = scipy.misc.imread(IMG_CONTENT)
        self.img_content = reshape_and_normalize_image(content_image, fixed_width, fixed_height)

        #img_style = scipy.misc.imread(IMG_STYLE)
        self.img_style = reshape_and_normalize_image(style_image, fixed_width, fixed_height)
        
        self.img_generated = generate_noise_image(self.img_content, fixed_width, fixed_height, fixed_channel)
        
        self.STYLE_LAYERS = [
                            ('conv1', 0.2),
                            ('conv2', 0.2),
                            ('conv3', 0.2),
                            ('conv4', 0.2),
                            ('conv5', 0.2)]

    def content_cost(self, con, gen):
        """
        Calculate the content cost of the generated picture.

        Parameters
        ----------
        con: array
             The content tensor image.

        gen: tensor
             The generated tensor image.
        """
        m, nw, nh, nc = gen.get_shape().as_list()
        con_flatten = tf.reshape(con, [-1])
        gen_flatten = tf.reshape(gen, [-1])
        J_content = tf.reduce_sum((con_flatten - gen_flatten)**2 / (4 * nw * nh * nc))
        return J_content
    
    def gram_matrix(self, A):
        """
        Calculate the gram matrix of a given input.

        Parameters
        ----------
        A: tensor
           The tensor for calculating the gram matrix.

        Reference
        ---------
        https://en.wikipedia.org/wiki/Gramian_matrix
        """
        return tf.matmul(A, tf.transpose(A))
    
    
    def style_cost(self, sty, gen):
        """
        Calculate the style cost of a single layer in the generated picture.

        Parameters
        ----------
        sty: array
             The style tensor image.

        gen: tensor
             The generated tensor image.
        """
        m, nw, nh, nc = gen.get_shape().as_list()
        sty_res = tf.transpose(tf.reshape(sty, (nw * nh, nc)))
        gen_res = tf.transpose(tf.reshape(gen, (nw * nh, nc)))
        gm_sty = self.gram_matrix(sty_res)
        gm_gen = self.gram_matrix(gen_res)
        sty_flatten = tf.reshape(gm_sty, [-1])
        gen_flatten = tf.reshape(gm_gen, [-1])
        J_style = tf.reduce_sum((sty_flatten - gen_flatten)**2 / (4 * nc**2 * (nh * nw)**2))
        return J_style
    
    def full_style_cost(self, model, sess):
        """
        Calculate the style cost of all layers in the generated picture.

        Parameters
        ----------
        model: dict
             The model we are using. Currently supports Alexnet

        sess: tensorflow session
             Session with current graph
        """
        J_style = 0
        STYLE_LAYERS = self.STYLE_LAYERS

        for layer_name, cost in STYLE_LAYERS:
            out = model[layer_name]
            a_S = sess.run(out)
            a_G = out

            J_style_layer = self.style_cost(a_S, a_G)
            J_style += cost * J_style_layer

        return J_style
    
    def total_cost(self, J_content, J_style, alpha = 10, beta = 40):
        """
        Calculate the total cost of neural style transfer.

        Parameters
        ----------
        J_content: float
            The content cost.

        J_style: float
            The style cost.

        alpha: double
            Hyperparameter. J_content weight.

        beta: double
            Hyperparameter. J_style weight.
        """
        J = alpha * J_content + beta * J_style
        return J
    
    def train(self, num_iter = 200):
        input_image = self.img_generated
        tf.reset_default_graph()
        model = export_graph(self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNEL)
        sess = tf.InteractiveSession()
        
        sess.run(model['input'].assign(self.img_content))
        out = model['conv4']
        a_C = sess.run(out)
        a_G = out
        J_content = self.content_cost(a_C, a_G)
        
        sess.run(model['input'].assign(self.img_style))
        J_style = self.full_style_cost(model, sess)
        
        J = self.total_cost(J_content, J_style)
        
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(J)
        
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))

        generated_images = []
        for i in range(num_iter):
            _ = sess.run(train_step)
            generated_image = sess.run(model['input'])

            if i % 20 == 0:
                jt, jc, js = sess.run([J, J_content, J_style])
                generated_images.append(generated_image)

        generated_images.append(generated_image)

        return generated_image
        
        
        
        
        
        
        
        
        
        
        
        
        
        