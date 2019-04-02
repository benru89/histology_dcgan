"""This module does blah blah."""
import os
import matplotlib.pyplot as plt
import numpy as np
from dcgan import generator
from constants import DIM_X, DIM_Y, DIM_Z, BASE_PATH, OUTPUT_PATH, BATCH_SIZE

FULL_OUTPUT_PATH = BASE_PATH + OUTPUT_PATH

def save_output(sess, z_batch_tensor, input_z, is_training, steps):
    """
    This function does blah blah.
    """
    example_z = sess.run(z_batch_tensor)
    samples, _ = sess.run(generator(input_z, 0.5, False),
                          feed_dict={input_z: example_z, is_training:False})
    imgs = [img[:, :, :] for img in samples]
    figure_side = 2
    fig, ax = plt.subplots(nrows=figure_side, ncols=figure_side, figsize=(40, 40))
    k = 0
    for i in range(figure_side):
        for j in range(figure_side):
            ax[i, j].imshow((np.reshape(imgs[k], (DIM_X, DIM_Y, DIM_Z))*255).astype(np.uint8))
            ax[i, j].axis('off')
            k = k+1
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    if not os.path.exists(FULL_OUTPUT_PATH):
        os.mkdir(FULL_OUTPUT_PATH)
    fig.savefig(FULL_OUTPUT_PATH + str(steps) + '_output.png')
    plt.close('all')
    with open(FULL_OUTPUT_PATH + str(steps)+ '_tensor.txt', 'w') as fout:
        for row in range(example_z.shape[0]):
            for col in range(example_z.shape[1]):
                fout.write(str(example_z[row, col]) + " ")

def save_single_output(sess, z_data, input_z, is_training, steps):
    """
    This function does blah blah.
    """
    samples, _ = sess.run(generator(input_z, 0.5, False),
                          feed_dict={input_z: z_data, is_training:False})
    imgs = [img[:, :, :] for img in samples]
    figure_side = int(np.sqrt(BATCH_SIZE))
    fig, ax = plt.subplots(nrows=figure_side, ncols=figure_side, figsize=(40, 40))
    k = 0
    for i in range(figure_side):
        for j in range(figure_side):
            ax[i, j].imshow((np.reshape(imgs[k], (DIM_X, DIM_Y, DIM_Z))*255).astype(np.uint8))
            ax[i, j].axis('off')
            k = k+1
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    if not os.path.exists(FULL_OUTPUT_PATH):
        os.mkdir(FULL_OUTPUT_PATH)
    fig.savefig(FULL_OUTPUT_PATH + str(steps) + '_output.png')
    with open(FULL_OUTPUT_PATH + str(steps)+ '_tensor.txt', 'w') as fout:
        for row in range(z_data.shape[0]):
            for col in range(z_data.shape[1]):
                fout.write(str(z_data[row, col]) + " ")
