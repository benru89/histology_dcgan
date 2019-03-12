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

# Generator
# Implement `generator` to generate an image using `z`. This function should be able to reuse the variables in the neural network. 
# Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name 
# of "generator" to allow the variables to be reused. 
# The function should return the generated 28 x 28 x `out_channel_dim` images.
def generator(z, keep_prob=0.5, is_training=True):
    activation = lrelu
    momentum = 0.9
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = z
        x = tf.layers.dense(x, units=4*4*1024, activation=activation)
        x = tf.reshape(x, shape=[-1,4,4,1024])
        print(x.shape)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
        print(x.shape)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        print(x.shape)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        print(x.shape)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        print(x.shape)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        logits = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=2, padding='same')
        print(x.shape)
        x = tf.tanh(logits)
        print(x.shape)
        return x, logits