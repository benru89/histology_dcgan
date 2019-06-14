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
from constants import (SEED, BATCH_SIZE, Z_NOISE_DIM, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS, Y_DIM,
                       BASE_PATH, DATA_PATH, D_LEARNING_RATE, G_LEARNING_RATE, BETA1, OUTPUT_PATH,
                       SAVE_MODEL_EVERY, PRINT_INFO_EVERY, CHKPTS_PATH, SAVE_EXAMPLE_EVERY, GRAPHS_PATH, WRITE_IMG_SUMMARY_EVERY)


def run():
    """
    run the program
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", nargs='?', const=1,
                        type=int, help="num of examples to generate")
    parser.add_argument("--labels", nargs='?', const='r',
                        help="p - positive labels, n - negative labels, no argument - random")
    parser.add_argument("--save_tensors", help="Save also tensors")
    parser.add_argument("--out_prefix", help="prefix for output filenames")

    parser.add_argument("--image", help="image_file")

    args = parser.parse_args()
    if args.out_prefix and (args.generate is None):
        parser.error("--out_prefix requires --generate argument")
    if args.save_tensors and (args.generate is None):
        parser.error("--save_tensors requires --generate argument")
    if args.labels and (args.generate is None):
        parser.error("--labels requires --generate argument")
    if args.out_prefix is None:
        args.out_prefix = "output"
    if args.save_tensors is None:
        args.save_tensors = False
    if args.labels == 'p':
        labels = [1] * BATCH_SIZE
        labels = np.eye(2)[labels]
    elif args.labels == 'n':
        labels = [0] * BATCH_SIZE
        labels = np.eye(2)[labels]
    else:
        labels = np.random.randint(0,2,BATCH_SIZE)
        labels = np.eye(2)[labels]

    # Seed for reproducibility
    tf.set_random_seed(SEED)

    # Dataset
    dataset, dataset_len = data.create_dataset(
        BASE_PATH + DATA_PATH, BATCH_SIZE, NUM_EPOCHS)
    iterator = dataset.make_initializable_iterator()

    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_global_step = tf.assign_add(
        global_step, 1, name='increment_global_step')

    # Model
    input_real, input_z, input_d_y, input_g_y = model.model_inputs(
        DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM, Y_DIM)
    total_steps = (dataset_len / BATCH_SIZE) * NUM_EPOCHS
    starter_stdev = 0.1

    ##decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    decaying_stdev = tf.train.exponential_decay(
        starter_stdev, global_step, total_steps * 10, 0.0001)

    decaying_noise = tf.random_normal(shape=tf.shape(
        input_real), mean=0.0, stddev=decaying_stdev, dtype=tf.float32)
    tf.summary.scalar("stdev", tf.keras.backend.std(
        decaying_noise), collections=["d_summ"])
    d_loss, g_loss = model.model_loss(
        input_real, input_z, input_d_y, input_g_y, decaying_noise=decaying_noise)
    d_train_opt, g_train_opt = model.model_opt(
        d_loss, g_loss, D_LEARNING_RATE, G_LEARNING_RATE, BETA1)

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
                sess, z_batch_tensor, input_z, input_g_y, labels, args.generate,
                save_tensor=args.save_tensors, name_prefix=args.out_prefix)

        elif args.image:
            start_img = Image.open(args.image)
            img, lat_vector = latent_space.search_image(sess, start_img)
            img.save(BASE_PATH + OUTPUT_PATH + 'rec_output.png')
            np.save(BASE_PATH + OUTPUT_PATH + "zp_rec", lat_vector)

        else:
            train(sess, saver, z_batch_tensor, increment_global_step, dataset_len,
                  iterator, input_real, input_z, input_d_y, input_g_y,
                  d_loss, g_loss, d_train_opt, g_train_opt)


def get_session(batch_size):
    """
      Returns an open tf session from last valid checkpoint. Useful to use a saved trained model.
      ie. use this function in a Jupyter notebook to create a session.
    """
    # Dataset
    dataset, dataset_len = data.create_dataset(
        BASE_PATH + DATA_PATH, BATCH_SIZE, NUM_EPOCHS)
    iterator = dataset.make_initializable_iterator()
 
    # Model
    input_real, input_z, input_d_y, input_g_y = model.model_inputs(
        DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM, Y_DIM)
    dcgan.generator(input_z, input_g_y)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=5)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    try:
        saver.restore(sess, tf.train.latest_checkpoint(
            BASE_PATH + CHKPTS_PATH))
        last_global_step = sess.run(global_step)
        print(last_global_step)
    except ValueError:
        print("Error loading checkpoint, no valid checkpoint found")

    return sess, input_real, input_z, input_g_y


def close_session(sess):
    sess.close()


def train(sess, saver, z_batch_tensor, increment_global_step, dataset_len, iterator, input_real, input_z, input_d_y, input_g_y, d_loss, g_loss, d_train_opt, g_train_opt):

    next_batch = iterator.get_next()

    writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
    g_imgs = tf.summary.merge_all(key="g_imgs")
    g_summ = tf.summary.merge_all(key="g_summ")
    d_summ = tf.summary.merge_all(key="d_summ")

    while True:
        try:
            steps = sess.run(increment_global_step)
            batch_imgs, batch_labels = sess.run(next_batch)

            batch_z = sess.run(z_batch_tensor)

            _, d_loss_sum_str = sess.run([d_train_opt, d_summ], feed_dict={
                                         input_real: batch_imgs, input_z: batch_z, input_d_y: batch_labels, input_g_y: batch_labels})
            writer.add_summary(d_loss_sum_str, steps)

            _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
                input_real: batch_imgs, input_z: batch_z, input_d_y: batch_labels, input_g_y: batch_labels})
            writer.add_summary(g_sum_str, steps)

            _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
                input_real: batch_imgs, input_z: batch_z, input_d_y: batch_labels, input_g_y: batch_labels})

            writer.add_summary(g_sum_str, steps)

            if steps % WRITE_IMG_SUMMARY_EVERY == 0:
                writer.add_summary(sess.run(g_imgs, feed_dict={
                                   input_z: batch_z, input_d_y: batch_labels, input_g_y: batch_labels}), steps)

            if steps % SAVE_MODEL_EVERY == 0:
                saver.save(sess, BASE_PATH + CHKPTS_PATH +
                           "model", global_step=steps)
            if steps % PRINT_INFO_EVERY == 0:
                train_loss_d, train_loss_g = sess.run([d_loss, g_loss], feed_dict={
                                                      input_real: batch_imgs, input_z: batch_z, input_d_y: batch_labels, input_g_y: batch_labels})

                print("Step {} de {}".format(steps % int(dataset_len/BATCH_SIZE)+1, int(dataset_len/BATCH_SIZE)+1),
                      "-- Epoch [{} de {}]".format(int(steps *
                                                       BATCH_SIZE/dataset_len), NUM_EPOCHS),
                      "-- Global step {}".format(steps),
                      "-- Discriminator Loss: {:.4f}".format(train_loss_d),
                      "-- Generator Loss: {:.4f}".format(train_loss_g))
            if steps % SAVE_EXAMPLE_EVERY == 0:
                #output.save_mosaic_output(sess, z_batch_tensor, input_z, input_d_y, input_g_y, steps)                
                output.save_output(sess, z_batch_tensor, input_z, input_g_y, batch_labels, steps, 3)
            gc.collect()
        except tf.errors.OutOfRangeError:
            print("End")
            break


if __name__ == '__main__':
    run()
