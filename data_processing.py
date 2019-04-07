"""This module does blah blah."""
from glob import glob
import os
from os.path import isfile, join
import tensorflow as tf
from PIL import Image
import numpy as np
from constants import SEED, NUM_THREADS

def extract_patches(image, patch_size, num_patches=10):
    """Get `num_patches` random crops from the image"""
    patches = []
    for i in range(num_patches):
        patch = tf.random_crop(image, [patch_size, patch_size, 3])
        patches.append(patch)

    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 3]
    return patches


def read_image(filename, channels):
    """
    This function does blah blah.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def train_preprocess(image):
    """
    This function does blah blah.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def create_dataset(path, batch_size, img_height, img_width, channels, num_epochs):
    """
    This function does blah blah.
    """
    
    convert_tiff_to_jpeg(path)
    filenames = glob(os.path.join(path, "*.jpg"))
    dataset = (tf.data.Dataset.from_tensor_slices((filenames))
                .repeat(num_epochs)
                .shuffle(buffer_size=len(filenames))
                .map(lambda filename: read_image(
                    filename, channels), num_parallel_calls=NUM_THREADS)
                .map(lambda image: extract_patches(
                    image, num_patches=8, patch_size=512), num_parallel_calls=NUM_THREADS)
                .map(train_preprocess, num_parallel_calls=NUM_THREADS)
                .map(lambda image: tf.image.resize_images(
                    image, [img_height, img_width]))
                .apply(tf.data.experimental.unbatch())
                .shuffle(buffer_size=len(filenames))
                .batch(batch_size)
                .prefetch(1))

    return dataset

def convert_tiff_to_jpeg(path):
    """
    This function does blah blah.
    """
    filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
    for filename in filenames:
        if os.path.splitext(filename)[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(path, filename))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % filename)
            else:
                outputfile = os.path.splitext(filename)[0] + ".jpg"
                try:
                    image = Image.open(os.path.join(path, filename))
                    print("Converting jpeg for %s" % filename)
                    image.thumbnail(image.size)
                    image.save(os.path.join(path, outputfile), "JPEG", quality=100)
                except IOError as err:
                    print("I/O error({0}): {1}".format(err.errno, err.strerror))
