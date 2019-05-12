"""This module does blah blah."""
import os
import matplotlib.pyplot as plt
import numpy as np
from dcgan_alt import sampler
from constants import DIM_X, DIM_Y, DIM_Z, BASE_PATH, OUTPUT_PATH, BATCH_SIZE, FULL_OUTPUT_PATH
from PIL import Image


def image_from_array(image_arr):
    return np.reshape(image_arr, (DIM_X, DIM_Y, DIM_Z))


def transform_image(image):
    # back from -1,1 to 0,1
    image = image / 2 + 0.5
    # from 0,1 to 0,255
    image = image * 255
    return image


def save_mosaic_output(sess, z_batch_tensor, input_z, steps, save_tensor=False):
    """
    This function does blah blah.
    """
    example_z = sess.run(z_batch_tensor)
    samples = sess.run(sampler(input_z),
                       feed_dict={input_z: example_z})
    imgs = [img[:, :, :] for img in samples]
    figure_side = int(np.sqrt(BATCH_SIZE))
    fig, ax = plt.subplots(nrows=figure_side, ncols=figure_side)
    k = 0
    for i in range(figure_side):
        for j in range(figure_side):
            ax[i, j].imshow(transform_image(
                image_from_array(imgs[k])).astype(np.uint8))
            ax[i, j].axis('off')
            k = k+1
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    if not os.path.exists(FULL_OUTPUT_PATH):
        os.mkdir(FULL_OUTPUT_PATH)
    fig.savefig(FULL_OUTPUT_PATH + str(steps) + '_mosaic.png')
    plt.close('all')
    if save_tensor:
        with open(FULL_OUTPUT_PATH + str(steps) + '_tensor.txt', 'w') as fout:
            for row in range(example_z.shape[0]):
                for col in range(example_z.shape[1]):
                    fout.write(str(example_z[row, col]) + " ")


def save_output(sess, z_batch_tensor, input_z, steps, num_samples=1, save_tensor=False):
    """
    This function does blah blah.
    """
    example_z = sess.run(z_batch_tensor)
    samples = sess.run(sampler(input_z),
                       feed_dict={input_z: example_z})
    imgs = [img[:, :, :] for img in samples]
    for i in range(num_samples):
        filepath = FULL_OUTPUT_PATH + str(steps) + '_output' + str(i) + '.png'
        output_image = Image.fromarray(transform_image(
            image_from_array(imgs[i])).astype(np.uint8))
        output_image.save(filepath)
    if save_tensor:
        for ex in range(example_z.shape[0]):
            with open(FULL_OUTPUT_PATH + str(steps) + str(ex) + '_tensor.txt', 'w') as fout:
                for col in range(example_z.shape[1]):
                    fout.write(str(example_z[ex, col]) + " ")


def generate_samples(sess, z_batch_tensor, input_z, steps, num_samples, save_tensor=False, name_prefix="output"):
    """
    This function does blah blah.
    """
    num_batches = num_samples // BATCH_SIZE + 1
    imgs_per_batch = BATCH_SIZE
    for i in range(num_batches):
        example_z = sess.run(z_batch_tensor)
        samples = sess.run(sampler(input_z), feed_dict={input_z: example_z})
        imgs = [img[:, :, :] for img in samples]
        if (i == num_batches):
            imgs_per_batch = num_samples % BATCH_SIZE
        for j in range(imgs_per_batch):
            filepath = FULL_OUTPUT_PATH + \
                str(steps) + name_prefix + str(i) + "_" + str(j) + '.png'
            output_image = Image.fromarray(transform_image(
                image_from_array(imgs[j])).astype(np.uint8))
            output_image.save(filepath)
        if save_tensor:
            for ex in range(example_z.shape[0]):
                filepath_txt = FULL_OUTPUT_PATH + \
                    str(steps) + name_prefix + str(i) + "_" + str(ex) + '.txt'
                np.savetxt(filepath_txt, example_z[ex], fmt='%1.25f')
