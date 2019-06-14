import tensorflow as tf
from dcgan_alt import discriminator, generator
from constants import BATCH_SIZE


def model_inputs(image_width, image_height, image_channels, z_dim, y_dim, batch_size = BATCH_SIZE):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    real_input_images = tf.placeholder(tf.float32, [batch_size] + [image_width, image_height, image_channels], name='real_images')
    input_z = tf.placeholder(tf.float32, [None, z_dim], name='input_z')
    g_labels = tf.placeholder(tf.float32, [None, y_dim], name='g_labels')
    d_labels = tf.placeholder(tf.float32, [None, y_dim], name='d_labels')

    return real_input_images, input_z, d_labels, g_labels

def model_loss(input_real, input_z, input_d_y, input_g_y, smooth_factor=0.1, decaying_noise=None):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    fake_samples = generator(input_z,input_g_y)
    tf.summary.image("G", fake_samples, max_outputs=2, collections=["g_imgs"])

    d_model_real, d_logits_real = discriminator(input_real, input_d_y, reuse=False, decaying_noise=decaying_noise)
    d_model_fake, d_logits_fake = discriminator(fake_samples, input_d_y, reuse=True, decaying_noise=decaying_noise)
        
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))    
    d_loss = d_loss_real + d_loss_fake
    

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, d_learning_rate, g_learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(d_learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    tf.summary.scalar("g_loss", g_loss, collections=["g_summ"])
    tf.summary.scalar("d_loss", d_loss, collections=["d_summ"])
    return d_train_opt, g_train_opt