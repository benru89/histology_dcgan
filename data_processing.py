"""This module does blah blah."""
from glob import glob
import os
from os.path import isfile, join
import tensorflow as tf
from PIL import Image
import numpy as np
from constants import SEED, NUM_THREADS

def read_normalize_resize_image(filename, img_height, img_width, channels):
    """
    This function does blah blah.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [img_height, img_width])
    return image


def train_preprocess(image):
    """
    This function does blah blah.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, np.random.randint(0, 3))
    #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def create_dataset(path, batch_size, img_height, img_width, channels, num_epochs):
    """
    This function does blah blah.
    """
    with tf.device('/cpu:0'):
        convert_tiff_to_jpeg(path)
        filenames = glob(os.path.join(path, "*.jpg"))
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.shuffle(buffer_size=len(filenames), seed=SEED)
        dataset = dataset.map(lambda filename: read_normalize_resize_image(
            filename, img_height, img_width, channels), num_parallel_calls=NUM_THREADS)
        dataset = dataset.map(train_preprocess, num_parallel_calls=NUM_THREADS)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
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
