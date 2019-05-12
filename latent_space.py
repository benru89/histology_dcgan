import tensorflow as tf
import numpy as np
from constants import Z_NOISE_DIM, BASE_PATH, OUTPUT_PATH, DIM_X, DIM_Y, DIM_Z, BATCH_SIZE, FULL_OUTPUT_PATH
from dcgan_alt import sampler
from PIL import Image
from output import transform_image, image_from_array

STEPS = 50

def search_image(sess, image_to_search):
    image_arr = (np.array(image_to_search) /127.5) - 1
    zp, zp_value = search(sess, image_arr)
    samples = sess.run(sampler(zp))
    imgs = [img[:, :, :] for img in samples]
    img = Image.fromarray(((np.reshape(imgs[0], (DIM_X, DIM_Y, DIM_Z)) + 1) * 127.5).astype(np.uint8))
    return img, zp_value

def search(sess,tensor, save_tensor=False):
    fz = tf.Variable(tensor, tf.float32)
    fz.initializer.run(session=sess)
    fz = tf.expand_dims(fz, 0)
    fz = tf.cast(fz,tf.float32)
    zp = tf.Variable(np.random.normal(size=(BATCH_SIZE,Z_NOISE_DIM)), dtype=tf.float32)
    zp.initializer.run(session=sess)
    fzp = sampler(zp)
   
    loss = tf.losses.mean_squared_error(labels=fz, predictions=fzp)

    # Decayed gradient descent
    descent_step = tf.Variable(0, trainable=False)
    descent_step.initializer.run(session=sess)
    starter_learning_rate = 0.99
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                            descent_step,
                                            10000, 0.00005)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # Optimize on the variable zp
    train = opt.minimize(loss, var_list=zp, global_step=descent_step)
    for i in range(1000): 
        _, loss_value, zp_val, eta = sess.run((train, loss, zp, learning_rate))
        print("%03d) eta=%03f, loss = %f" % (i, eta, loss_value))
        
    zp_val = sess.run(zp)
    print(zp_val.shape)
    print(zp_val * 0.5)
    
    return zp, zp_val

def interpolate(sess, tensor_a, tensor_b):
    z = tf.Variable(np.empty(shape=(STEPS, Z_NOISE_DIM)), dtype=tf.float32)
    for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=STEPS)):
       z[i] = (1 - alpha) * tensor_a + alpha * tensor_b

    fzp = sampler(z)
    samples = sess.run(fzp)
    imgs = [img[:, :, :] for img in samples]
    for i, image in enumerate(imgs):
        filepath = FULL_OUTPUT_PATH + 'interp_' + str(i) + '.png'
        output_image = Image.fromarray(transform_image(image_from_array(image)).astype(np.uint8))
        output_image.save(filepath)

def interpolate_images(sess, image_a, image_b):
    tensor_a, _ = search_image(sess, image_a)
    tensor_b, _ = search_image(sess, image_b)
    interpolate(sess, tensor_a, tensor_b)

def interpolate_rand(sess, image):
    tensor_a, _ = search_image(sess, image)
    rand_tensor = tf.random.uniform((1, Z_NOISE_DIM), dtype=tf.float32, minval=-1, maxval=1)
    interpolate(sess, tensor_a, rand_tensor)
