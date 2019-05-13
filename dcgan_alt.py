import math
import tensorflow as tf
from constants import BATCH_SIZE, DIM_X, DIM_Y, DIM_Z


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[
            1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def discriminator(image, reuse=False):
    df_dim = 64
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4


def generator(z):
    gf_dim = 64
    with tf.variable_scope("generator") as scope:

        s_h, s_w = DIM_X, DIM_Y
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        z_, h0_w, h0_b = linear(z, gf_dim*8*s_h16*s_w16,
                                'g_h0_lin', with_w=True)
        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0))
        h1, h1_w, h1_b = deconv2d(
            h0, [BATCH_SIZE, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))
        h2, h2_w, h2_b = deconv2d(
            h1, [BATCH_SIZE, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))
        h3, h3_w, h3_b = deconv2d(
            h2, [BATCH_SIZE, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))
        h4, h4_w, h4_b = deconv2d(
            h3, [BATCH_SIZE, s_h, s_w, DIM_Z], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)


def sampler(z, batch_size=BATCH_SIZE):
    gf_dim = 64
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = DIM_X, DIM_Y
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        h0 = tf.reshape(
            linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0, train=False))

        h1 = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1, train=False))

        h2 = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=False))

        h3 = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3, train=False))

        h4 = deconv2d(h3, [batch_size, s_h, s_w, DIM_Z], name='g_h4')

    return tf.nn.tanh(h4)
