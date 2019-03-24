import tensorflow as tf
from glob import glob
import os
from constants import SEED, NUM_THREADS

def read_normalize_resize_image(filename, img_height, img_width, channels):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [img_height, img_width])
    
    return image

def train_preprocess(image):
    #image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

def create_dataset(path, batch_size, img_height, img_width, channels, num_epochs):
    with tf.device('/cpu:0'):
        filenames = glob(os.path.join(path, "*.jpg"))
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.shuffle(buffer_size=len(filenames),seed = SEED)
        dataset = dataset.map(lambda filename: read_normalize_resize_image(filename, img_height, img_width, channels), num_parallel_calls = NUM_THREADS)
        dataset = dataset.map(train_preprocess, num_parallel_calls = NUM_THREADS)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset