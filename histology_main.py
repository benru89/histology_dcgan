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
                       PRINT_INFO_EVERY, CHKPTS_PATH, SAVE_EXAMPLE_EVERY)
import matplotlib.pyplot as plt

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
        z_batch_tensor = tf.random.normal((BATCH_SIZE, Z_NOISE_DIM), dtype=tf.float32)

        # Dataset
        dataset = data.create_dataset(BASE_PATH + DATA_PATH, BATCH_SIZE, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        # Model
        input_real, input_z, _ = model.model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM)
        is_training = tf.Variable(True, name='is_training')
        d_loss, g_loss = model.model_loss(input_real, input_z, DIM_Z, is_training)
        d_train_opt, g_train_opt = model.model_opt(d_loss, g_loss, D_LEARNING_RATE, G_LEARNING_RATE, BETA1)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
        # names_to_vars = {v.op.name: v for v in tf.global_variables()}
        # del names_to_vars["is_training"]
        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

        with tf.Session() as sess:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(BASE_PATH + CHKPTS_PATH))
                last_global_step = sess.run(global_step)
                print(last_global_step)
            except:
                sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            while True:
                try:
                    # Run optimizers
                    batch = sess.run(next_batch)
                    #uncomment to see how the images in the batch look like
                    """imgs = [img[:, :, :] for img in batch]
                    figure_side = 8
                    fig, ax = plt.subplots(nrows=figure_side, ncols=figure_side, figsize=(40, 40))
                    k = 0
                    for i in range(figure_side):
                        for j in range(figure_side):
                            ax[i, j].imshow((np.reshape(imgs[k], (DIM_X, DIM_Y, DIM_Z))*255).astype(np.uint8))
                            ax[i, j].axis('off')
                            k = k+1
                    fig.subplots_adjust(hspace=0.01, wspace=0.01)
                    plt.show()"""
                    steps = sess.run(global_step)
                    #when a noise file is specified
                    if args.z_file is not None:
                        z_file = open(BASE_PATH + OUTPUT_PATH + args.z_file, "r")
                        z_file_data = z_file.read().strip()
                        z_data = z_file_data.split(" ")
                        z_data = np.array(z_data).astype(np.float).reshape(BATCH_SIZE, Z_NOISE_DIM)
                        output.save_single_output(sess, z_data, input_z, is_training, steps)
                        break
                    #otherwise training mode
                    else:
                        batch_z = sess.run(z_batch_tensor)

                    sess.run(d_train_opt, feed_dict={input_real: batch, input_z: batch_z, is_training : True})
                    sess.run(g_train_opt, feed_dict={input_z: batch_z, is_training : True})
                    sess.run(increment_global_step)
                    #writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
                    if steps % SAVE_MODEL_EVERY == 0:
                        saver.save(sess, BASE_PATH + CHKPTS_PATH + "model", global_step=global_step)
                    if steps % PRINT_INFO_EVERY == 0:
                        train_loss_d = d_loss.eval({input_real: batch, input_z: batch_z, is_training : True}, session=sess)
                        train_loss_g = g_loss.eval({input_z: batch_z, is_training : True}, session=sess)
                        gc.collect()
                        print("Epoch Step {}...".format(steps),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                    if steps % SAVE_EXAMPLE_EVERY == 0:
                        output.save_output(sess, z_batch_tensor, input_z, is_training, steps)
                except tf.errors.OutOfRangeError:
                    print("End")
                    break

if __name__ == '__main__':
    run()
    