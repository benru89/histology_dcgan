import tensorflow as tf
import data_processing as data
import model
import numpy as np
import output
from constants import *
import argparse
import configparser


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--z_file", help="z_noise_file")
    args = parser.parse_args()

    with tf.Graph().as_default() as graph1:
        # Random seed for reproducibility
        tf.set_random_seed(SEED)
        
        # Placeholders
        z_batch_tensor = tf.random.normal((BATCH_SIZE, Z_NOISE_DIM),0, 2, dtype=tf.float32)

        # Dataset
        dataset = data.create_dataset(BASE_PATH + DATA_PATH, BATCH_SIZE, DIM_X, DIM_Y, DIM_Z, NUM_EPOCHS)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        saveable_obj = tf.data.experimental.make_saveable_from_iterator(iterator)

        # Model
        input_real, input_z, _ = model.model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        d_loss, g_loss = model.model_loss(input_real, input_z, DIM_Z, is_training)
        d_train_opt, g_train_opt = model.model_opt(d_loss, g_loss, LEARNING_RATE, BETA1)

        steps = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, BASE_PATH + CHKPTS_PATH + "model.ckpt")
            except:
                sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            while True:
                steps += 1
                try:
                    # Run optimizers
                    batch = sess.run(next_batch)
                    #when a noise file is specified
                    if (args.z_file != None):
                        z_file = open(BASE_PATH + OUTPUT_PATH + args.z_file,"r")
                        z_file_data = z_file.read().strip()
                        z_data = z_file_data.split(" ")
                        z_data = np.array(z_data).astype(np.float).reshape(BATCH_SIZE,Z_NOISE_DIM)
                        output.save_single_output(sess,z_data,input_z,is_training,steps)
                        break
                    #otherwise training mode
                    else:
                        batch_z = sess.run(z_batch_tensor)
                    sess.run(d_train_opt, feed_dict={input_real: batch, input_z: batch_z,  is_training : True})
                    sess.run(g_train_opt, feed_dict={input_z: batch_z, is_training : True})
                    #writer = tf.summary.FileWriter(BASE_PATH + GRAPHS_PATH, sess.graph)
                    if steps % SAVE_MODEL_EVERY == 0:
                        saver.save(sess, BASE_PATH + CHKPTS_PATH + "model.ckpt")
                    if steps % PRINT_INFO_EVERY == 0:
                        train_loss_d = d_loss.eval({input_real: batch, input_z: batch_z,is_training : True}, session = sess)
                        train_loss_g = g_loss.eval({input_z: batch_z,is_training : True}, session = sess)
                        print("Epoch Step {}...".format(steps),"Discriminator Loss: {:.4f}...".format(train_loss_d),"Generator Loss: {:.4f}".format(train_loss_g))
                    if steps % SAVE_EXAMPLE_EVERY == 0:
                        output.save_output(sess,z_batch_tensor,input_z,is_training,steps)
                        
                except tf.errors.OutOfRangeError:
                    print("End")
                    break