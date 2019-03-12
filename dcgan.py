import tensorflow as tf

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def discriminator(img_in, reuse=None, keep_prob=0.5):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 128,128,3])
        print(x.shape)
        x = tf.layers.conv2d(x, kernel_size=5, filters=32, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        print(x.shape)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        print(x.shape)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        print(x.shape)
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        print(x.shape)
        x = tf.layers.conv2d(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        print(x.shape)
        x = tf.contrib.layers.flatten(x)
        logits = tf.layers.dense(x, units=1)
        print(x.shape)
        x = tf.sigmoid(logits)
        print(x.shape)
        return x , logits

