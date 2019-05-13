"""This module does blah blah."""
import argparse
import gc
import tensorflow as tf
import numpy as np
from PIL import Image
import data_processing as data
import model
import output
import dcgan_alt as dcgan
import latent_space
from constants import (SEED, BATCH_SIZE, Z_NOISE_DIM, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS,
                       BASE_PATH, DATA_PATH, D_LEARNING_RATE, G_LEARNING_RATE, BETA1, OUTPUT_PATH,
                       SAVE_MODEL_EVERY, PRINT_INFO_EVERY, CHKPTS_PATH, SAVE_EXAMPLE_EVERY, GRAPHS_PATH)


def run():
    """
    run the program
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", nargs='?', const=1,
                        type=int, help="num of examples to generate")
    parser.add_argument("--image", help="image_file")
    parser.add_argument("--out_prefix", help="prefix for output filenames")
    args = parser.parse_args()
    if args.out_prefix and (args.generate is None):
        parser.error("--out_prefix requires --generate")
    if args.out_prefix == None:
      args.out_prefix = "output"
      
    # Seed for reproducibility
    tf.set_random_seed(SEED)

    # Dataset
    dataset, dataset_len = data.create_dataset(
        BASE_PATH + DATA_PATH, BATCH_SIZE, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS)
    iterator = dataset.make_initializable_iterator()

    # Model
    input_real, input_z = model.model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM)
    d_loss, g_loss = model.model_loss(input_real, input_z)
    d_train_opt, g_train_opt = model.model_opt(
        d_loss, g_loss, D_LEARNING_RATE, G_LEARNING_RATE, BETA1)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_global_step = tf.assign_add(
        global_step, 1, name='increment_global_step')
    z_batch_tensor = tf.random.uniform(
        (BATCH_SIZE, Z_NOISE_DIM), dtype=tf.float32, minval=-1, maxval=1)

    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(
                BASE_PATH + CHKPTS_PATH))
            last_global_step = sess.run(global_step)
            print(last_global_step)
        except ValueError:
            print("Error loading checkpoint, no valid checkpoint found")

        if args.generate:
            output.generate_samples(
                sess, z_batch_tensor, input_z, args.generate, save_tensor=True, name_prefix=args.out_prefix)

        elif args.image:
            start_img = Image.open(args.image)
            img, lat_vector = latent_space.search_image(sess, start_img)
            img.save(BASE_PATH + OUTPUT_PATH + 'rec_output.png')
            np.save(BASE_PATH + OUTPUT_PATH + "zp_rec", lat_vector)

        else:
            train(sess, saver, z_batch_tensor, increment_global_step, dataset_len,
                  iterator, input_real, input_z, d_loss, g_loss, d_train_opt, g_train_opt)


def get_session(batch_size):
    """
      Returns an open tf session from last valid checkpoint. Useful to use a saved trained model.
      ie. use this function in a Jupyter notebook to create a session.
    """
    #Dataset
    dataset, dataset_len = data.create_dataset(BASE_PATH + DATA_PATH, BATCH_SIZE, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS)
    iterator = dataset.make_initializable_iterator()
   
    #Model
    input_real, input_z = model.model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM, batch_size)
    dcgan.generator(input_z)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    try:
        saver.restore(sess, tf.train.latest_checkpoint(
            BASE_PATH + CHKPTS_PATH))
        last_global_step = sess.run(global_step)
        print(last_global_step)
    except ValueError:
        print("Error loading checkpoint, no valid checkpoint found")

    return sess, input_real, input_z


def close_session(sess):
    sess.close()


def train(sess, saver, z_batch_tensor, increment_global_step, dataset_len, iterator, input_real, input_z, d_loss, g_loss, d_train_opt, g_train_opt):

    next_batch = iterator.get_next()

    writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
    g_summ = tf.summary.merge_all(key="g_summ")
    d_summ = tf.summary.merge_all(key="d_summ")

    while True:
        try:
            steps = sess.run(increment_global_step)
            batch = sess.run(next_batch)
            batch_z = sess.run(z_batch_tensor)

            _, d_loss_sum_str = sess.run([d_train_opt, d_summ], feed_dict={
                                         input_real: batch, input_z: batch_z})
            writer.add_summary(d_loss_sum_str, steps)
            _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
                                    input_real: batch, input_z: batch_z})
            writer.add_summary(g_sum_str, steps)
            _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
                                    input_real: batch, input_z: batch_z})
            writer.add_summary(g_sum_str, steps)

            if steps % SAVE_MODEL_EVERY == 0:
                saver.save(sess, BASE_PATH + CHKPTS_PATH +
                           "model", global_step=steps)
            if steps % PRINT_INFO_EVERY == 0:
                train_loss_d, train_loss_g = sess.run([d_loss, g_loss], feed_dict={
                                                      input_real: batch, input_z: batch_z})

                print("Step {} de {}".format(steps % int(dataset_len/BATCH_SIZE)+1, int(dataset_len/BATCH_SIZE)+1),
                      "-- Epoch [{} de {}]".format(int(steps *
                                                       BATCH_SIZE/dataset_len), NUM_EPOCHS),
                      "-- Global step {}".format(steps),
                      "-- Discriminator Loss: {:.4f}".format(train_loss_d),
                      "-- Generator Loss: {:.4f}".format(train_loss_g))
            if steps % SAVE_EXAMPLE_EVERY == 0:
                output.save_mosaic_output(sess, z_batch_tensor, input_z, steps)
                output.save_output(sess, z_batch_tensor,
                                   input_z, steps, num_samples=3)
            gc.collect()
        except tf.errors.OutOfRangeError:
            print("End")
            break


if __name__ == '__main__':
    run()
