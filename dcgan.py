import tensorflow as tf

def discriminator(img_in,keep_prob=0.5):
    activation = lambda x: tf.nn.leaky_relu(x, alpha=0.02)
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        x = tf.reshape(img_in, shape=[-1, 128, 128, 3])
        x = tf.layers.conv2d(x, kernel_size=5, filters=32, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        logits = tf.layers.dense(x, units=1)
        x = tf.sigmoid(logits)
        return x, logits

def generator(z, keep_prob=0.5, is_training=True):
    activation = tf.nn.relu
    momentum = 0.9
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = z
        x = tf.layers.dense(x, units=4*4*1024, activation=activation)
        x = tf.reshape(x, shape=[-1, 4, 4, 1024])
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        logits = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=2, padding='same')
        x = tf.tanh(logits)
        return x, logits