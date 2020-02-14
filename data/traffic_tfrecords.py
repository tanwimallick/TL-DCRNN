#!/usr/bin/env python3
"""MNIST record generator"""

#import argparse
import os
import sys
import six
import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#from data import load_mnist_data

def _int64_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Creates a tf.Train.Feature from a bytes value."""
    if value is None:
        value = []
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _data_path(data_dir, name):
    """Constructs a full path to a TFRecord file to be stored in the
    data_dir. Will also ensure that the data directory exists.

    Args:
        data_dir (str):
            The directory where the records will be stored
        name (str):
            The name of the TFRecord

    Returns:
        (str):
            The full path to the TFRecord file
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, '%s.tfrecords' % name)


def write_as_tfrecords(x_data, y_data, name, data_dir, num_shards=1):
    """Serializes a dataset of images and labels and writes
    it as a sequence of binary strings in TFRecord format.

    Args:
        images (np.array):
            Array of shape (num_examples, width, height, num_channels)
            with set of input images.
        labels (np.array):
            Array of shape (num_examples, 1) with set of labels.
        name (str):
            The name of the data set
            (i.e. prefix to use for TFRecord names)
        data_dir (str):
            The directory where records will be stored
        num_shards (int):
            The number of files on disk to separate records into
    """
    print('Processing %s data' % name)
    num_examples = x_data.shape[0]

    def _process_examples(start_idx, end_index, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(
                    "\rProcessing sample %i of %i" % (index + 1, num_examples))
                sys.stdout.flush()

                x_row = x_data[index].tostring()
                y_row = y_data[index].tostring()
                x_shape = np.array(x_data[index].shape, np.int64).tostring()
                y_shape = np.array(y_data[index].shape, np.int64).tostring()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "x_shape": _bytes_feature(x_shape),
                        "y_shape": _bytes_feature(y_shape),
                        'x': _bytes_feature(x_row),
                        'y': _bytes_feature(y_row)
                    }))
                writer.write(example.SerializeToString())

    samples_per_shard = num_examples // num_shards


    for shard in range(num_shards):
        start_index = shard * samples_per_shard
        end_index = start_index + samples_per_shard
        _process_examples(
            start_index,
            end_index,
            _data_path(data_dir, '%s' % (name)))

    print("")

def _parse_record_fn(data_record):
    features = {
        'x_shape': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'y_shape': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'x': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'y': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
    }
    sample = tf.parse_single_example(data_record, features)

    x_shape = tf.decode_raw(sample['x_shape'], tf.int64)
    y_shape = tf.decode_raw(sample['y_shape'], tf.int64)
    x = tf.decode_raw(sample['x'], tf.float64)
    y = tf.decode_raw(sample['y'], tf.float64)
    x = tf.reshape(x, shape=x_shape)
    y = tf.reshape(y, shape=y_shape)
    return x, y

 
'''
def get_arguments():
    """Creates args dictionary from command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--outdir',
        default='./data_dir/mnist',
        help='directory where TFRecords will be stored')
    return parser.parse_args()


def main():
    """Serializes the MNIST dataset (Yann LeCun and Corinna Cortes)
    http://yann.lecun.com/exdb/mnist/ and writes it into disk in
    the recommended TFRecord format.

    The dataset created by this script will be compatible with the
    input function defined in data.py, which will be used as part
    of an efficient input pipeline when training the MNIST digit
    classifiers defined in tf_model.py and hybrid_model.py using
    the Estimator API.
    """
    args = get_arguments()

    data_dir = os.path.expanduser(args.outdir)

    x_train, y_train, x_test, y_test = load_mnist_data(one_hot_labels=False)
    write_as_tfrecords(
        x_train, y_train, 'train', data_dir, num_shards=10)
    write_as_tfrecords(
        x_test, y_test , 'test', data_dir)

    print('\ntfrecords written to %s.\n' % args.outdir)


def _int64_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def _float_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:  
        raise ValueError("The input should be numpy ndarray. \
                           Instaed got {}".format(ndarray.dtype))


def _float_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
'''



