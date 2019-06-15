import tensorflow as tf
import numpy as np
from constants import Z_NOISE_DIM, BASE_PATH, OUTPUT_PATH, DIM_X, DIM_Y, DIM_Z, Y_DIM, BATCH_SIZE, FULL_OUTPUT_PATH
from dcgan_alt import sampler
from PIL import Image
from output import transform_image, image_from_array


def search_image(sess, image_to_search,decay_steps):
    image_arr = (np.array(image_to_search) / 127.5) - 1
    
    labels = np.random.randint(0,1,1)
    labels_one_hot = tf.one_hot(labels, Y_DIM, 1.0, 0.0)  
    zp, zp_value = search(sess, image_arr,labels_one_hot,decay_steps)
    
   
    samples = sess.run(sampler(zp,labels_one_hot,1))
    imgs = [img[:, :, :] for img in samples]
    img = Image.fromarray(
        ((np.reshape(imgs[0], (DIM_X, DIM_Y, DIM_Z)) + 1) * 127.5).astype(np.uint8))
    return img, zp_value


def search(sess, tensor, labels_one_hot,decay_steps,save_tensor=False):
    fz = tf.Variable(tensor, tf.float32)
    fz.initializer.run(session=sess)
    fz = tf.expand_dims(fz, 0)
    fz = tf.cast(fz, tf.float32)
    zp = tf.Variable(np.random.normal(
        size=(1, Z_NOISE_DIM)), dtype=tf.float32)
    zp.initializer.run(session=sess)
    fzp = sampler(zp,labels_one_hot,1)

    loss = tf.losses.mean_squared_error(labels=fz, predictions=fzp)

    # Decayed gradient descent
    descent_step = tf.Variable(0, trainable=False)
    descent_step.initializer.run(session=sess)
    starter_learning_rate = 0.99
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               descent_step,
                                               decay_steps, 0.05)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # Optimize on the variable zp
    train = opt.minimize(loss, var_list=zp, global_step=descent_step)
    for i in range(decay_steps):
        _, loss_value, zp_val, eta = sess.run((train, loss, zp, learning_rate))
        print("%03d) eta=%03f, loss = %f" % (i, eta, loss_value))

    zp_val = sess.run(zp)
    return zp, zp_val
  
def interpolate(sess, tensor_a, tensor_b, labels, steps=50, prefix=None):
    z = np.empty(shape=(steps, Z_NOISE_DIM))
    for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=steps)):
        z[i] = (1-alpha) * tensor_a + alpha * tensor_b

    z_ = tf.placeholder(tf.float32, [steps, Z_NOISE_DIM])
    g_labels = tf.placeholder(tf.float32, [None, Y_DIM])
    labels = [labels[0]] * int(steps/2) + [labels[1]] * int(steps/2)
    labels = np.eye(2)[labels]
    samples = sess.run(sampler(z_,g_labels, steps), feed_dict={z_: z, g_labels:labels})
    imgs = [img[:, :, :] for img in samples]
    for i, image in enumerate(imgs):
        filepath = FULL_OUTPUT_PATH + 'interp_'
        if prefix is not None:
          filepath += prefix
        filepath += str(i) + '.png'
        
        
        output_image = Image.fromarray(transform_image(
            image_from_array(image)).astype(np.uint8))
        output_image.save(filepath)


def interpolate_images(sess, image_a, image_b, labels, steps=50, prefix=None):
    tensor_a, _ = search_image(sess, image_a)
    tensor_b, _ = search_image(sess, image_b)
    interpolate(sess, tensor_a, tensor_b, labels, steps)


def interpolate_rand(sess, image, steps=50):
    tensor_a, _ = search_image(sess, image)
    rand_tensor = tf.random.uniform(
        (1, Z_NOISE_DIM), dtype=tf.float32, minval=-1, maxval=1)
    interpolate(sess, tensor_a, rand_tensor, steps)


def add_tensors(tensor_a, tensor_b):
    return tensor_a + tensor_b

def substract_tensors(tensor_a, tensor_b):
    return tensor_a - tensor_b
  
def average_tensor(tensors):
  return np.mean(tensors, axis=0)  