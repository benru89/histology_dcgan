import tensorflow as tf
from constants import BATCH_SIZE, DIM_X, DIM_Y, DIM_Z


def discriminator(img_in, reuse=False, training=True):
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 256, 256, 3])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, kernel_size=6, filters=64, strides=2, padding='same'))
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=training), 0.02)
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=training), 0.02)
        x = tf.layers.conv2d(x, kernel_size=5, filters=512, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=training), 0.02)
        x = tf.contrib.layers.flatten(x)
        logits = tf.layers.dense(x, units=1)
        x = tf.sigmoid(logits)
        return x, logits


def generator(z, training=True):
    momentum = 0.9
    epsilon = 1e-5
    with tf.variable_scope("generator", reuse=(not training)):
        x = tf.layers.dense(z, units=16*16*512, activation=tf.nn.relu)
        x = tf.reshape(x, shape=[-1, 16, 16, 512])
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training, momentum=momentum, epsilon=epsilon))
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training, momentum=momentum, epsilon=epsilon))
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training, momentum=momentum, epsilon=epsilon))
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training, momentum=momentum, epsilon=epsilon))
        logits = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=2, padding='same')
        x = tf.tanh(logits)
        return x, logits
