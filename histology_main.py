"""This module does blah blah."""
import argparse
import tensorflow as tf
import numpy as np
import data_processing as data
import model
import gc
import output
from constants import (SEED, BATCH_SIZE, Z_NOISE_DIM, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS,
                       BASE_PATH, DATA_PATH, D_LEARNING_RATE, G_LEARNING_RATE, BETA1, OUTPUT_PATH, SAVE_MODEL_EVERY,
                       PRINT_INFO_EVERY, CHKPTS_PATH, SAVE_EXAMPLE_EVERY, GRAPHS_PATH)
import matplotlib.pyplot as plt
from dcgan import generator

def run():
    """
    run the program
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--z_file", help="z_noise_file")
    args = parser.parse_args()

    with tf.Graph().as_default():
        # Random seed for reproducibility
        tf.set_random_seed(SEED)

        # Placeholders
        z_batch_tensor = tf.random.uniform((BATCH_SIZE, Z_NOISE_DIM), dtype=tf.float32, minval=-1, maxval=1)

        # Dataset
        dataset, dataset_len = data.create_dataset(BASE_PATH + DATA_PATH, BATCH_SIZE, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        # Model
        input_real, input_z = model.model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM)
        d_loss, g_loss = model.model_loss(input_real, input_z, DIM_Z)
        d_train_opt, g_train_opt = model.model_opt(d_loss, g_loss, D_LEARNING_RATE, G_LEARNING_RATE, BETA1)
        
        global_step = tf.Variable(0, trainable=False, name='global_step')
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, tf.train.latest_checkpoint(BASE_PATH + CHKPTS_PATH))
                last_global_step = sess.run(global_step)
                print(last_global_step)
            except:
                print("Error loading checkpoint")
                
            sess.run(iterator.initializer)
            writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
            g_loss_sum = tf.summary.scalar("g_loss", g_loss)
            d_loss_sum = tf.summary.scalar("d_loss", d_loss)
            samples, _ = generator(input_z, training=False)
            g_img_sum = tf.summary.image("G", samples, max_outputs=10)
            g_sum = tf.summary.merge([g_loss_sum, g_img_sum])
            while True:
                try:
                    batch = sess.run(next_batch)
                    steps = sess.run(global_step)
                    batch_z = sess.run(z_batch_tensor)

                    _, d_loss_sum_str = sess.run([d_train_opt, d_loss_sum], feed_dict={input_real: batch, input_z: batch_z})
                    writer.add_summary(d_loss_sum_str, steps)
                    _, g_sum_str = sess.run([g_train_opt, g_sum], feed_dict={input_real: batch,input_z: batch_z})
                    writer.add_summary(g_sum_str, steps)
                    _, g_sum_str = sess.run([g_train_opt, g_sum], feed_dict={input_real: batch,input_z: batch_z})
                    writer.add_summary(g_sum_str, steps)
                    sess.run(increment_global_step)
                               
                    if steps % SAVE_MODEL_EVERY == 0:
                        saver.save(sess, BASE_PATH + CHKPTS_PATH + "model", global_step=global_step)
                    if steps % PRINT_INFO_EVERY == 0:
                        train_loss_d, train_loss_g = sess.run([d_loss, g_loss],feed_dict={input_real: batch, input_z: batch_z})
                        print("Step {} de {}".format(steps%int(dataset_len/BATCH_SIZE)+1, int(dataset_len/BATCH_SIZE)+1),
                              "-- Epoch [{} de {}]".format(int(steps * BATCH_SIZE/dataset_len),NUM_EPOCHS),
                              "-- Global step {}".format(steps),
                              "-- Discriminator Loss: {:.4f}".format(train_loss_d),
                              "-- Generator Loss: {:.4f}".format(train_loss_g))
                    if steps % SAVE_EXAMPLE_EVERY == 0:
                        output.save_mosaic_output(sess, z_batch_tensor, input_z, steps)
                    gc.collect()
                except tf.errors.OutOfRangeError:
                    print("End")
                    break

if __name__ == '__main__':
    run()
    