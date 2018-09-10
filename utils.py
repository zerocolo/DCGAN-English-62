from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
      crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
       cropped_image = center_crop(
         image, input_height, input_width,
         resize_height, resize_width)
    else:
       cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])

    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def visualize(sess, dcgan, labels,  config, option):

    image_frame_dim = int(math.ceil(config.batch_size**.5))

    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
        print(" [*] %d" % idx)
        z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))

        for kdx, z in enumerate(z_sample):
            z[idx] = values[kdx]

        if option:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        else:
            y = []
            j = 0
            for i in range(64):
                y.append(labels[j])
                j += 1
                if j == len(labels):
                    j = 0

            y_one_hot = np.zeros((config.batch_size, 62))
            y_one_hot[np.arange(config.batch_size), y] = 1

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))

def image_manifold_size(num_images):
     manifold_h = int(np.floor(np.sqrt(num_images)))
     manifold_w = int(np.ceil(np.sqrt(num_images)))
     assert manifold_h * manifold_w == num_images
     return manifold_h, manifold_w