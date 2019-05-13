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


def save_images(imgs_batch, curr_batch=1, stepcount=None, name_prefix=None):
    for i, image in enumerate(imgs_batch):
        filepath = FULL_OUTPUT_PATH
        if stepcount:
            filepath += str(stepcount)
        elif name_prefix:
            filepath += str(name_prefix)
        filepath += "_" + str(curr_batch) + "_" + str(i) + '.png'
        output_image = Image.fromarray(transform_image(
            image_from_array(image)).astype(np.uint8))
        output_image.save(filepath)


def save_tensors(z_batch, curr_batch=1, stepcount=None, name_prefix=None):
    for i, ex in enumerate(z_batch):
        filepath = FULL_OUTPUT_PATH
        if stepcount:
            filepath += str(stepcount)
        elif name_prefix:
            filepath += str(name_prefix)
        filepath += "_" + str(curr_batch) + "_" + str(i) + '.txt'
        np.savetxt(filepath, ex, fmt='%1.25f')


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


def save_output(sess, z_batch_tensor, input_z, steps, num_samples=1, save_tensor=False):
    """
    This function does blah blah.
    """
    example_z = sess.run(z_batch_tensor)
    samples = sess.run(sampler(input_z),
                       feed_dict={input_z: example_z})
    imgs = [img[:, :, :] for img in samples]
    if num_samples > BATCH_SIZE:
        num_samples = BATCH_SIZE
    save_images(imgs, num_samples, stepcount=steps)
    if save_tensor:
        save_tensors(example_z)


def generate_samples(sess, z_batch_tensor, input_z, num_samples, save_tensor=False, name_prefix="output"):
    """
    This function does blah blah.
    """
    num_batches = num_samples // BATCH_SIZE
    for batch_count in range(num_batches):
        example_z = sess.run(z_batch_tensor)
        samples = sess.run(sampler(input_z), feed_dict={input_z: example_z})
        imgs = [img[:, :, :] for img in samples]
        save_images(imgs, batch_count, name_prefix=name_prefix)
        if save_tensor:
            save_tensors(example_z, batch_count, name_prefix=name_prefix)
