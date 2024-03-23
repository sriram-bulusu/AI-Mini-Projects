import numpy as np
import scipy.io as sio
import os
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# TF2 doesn't include the mnist module in tensorflow.examples.tutorials
# Use tf.keras.datasets.mnist instead
def load_data(name, random_labels=False):
    """Load the data
    name - the name of the dataset
    random_labels - True if we want to return random labels to the dataset
    return object with data and labels"""
    print('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()
    if name.split('/')[-1] == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        data_sets.data = np.concatenate((x_train, x_test), axis=0).reshape((-1, 28*28)) / 255.0  # Normalize
        data_sets.labels = np.concatenate((tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)), axis=0)
        # data_sets.labels = np.eye(10, dtype='float32')[np.concatenate((y_train, y_test), axis=0)]
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print("Shapes:")
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        data_sets.data = np.concatenate((x_train, x_test), axis=0).reshape((-1, 32*32*3))
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        train_labels = np.eye(10, dtype='float32')[y_train]
        test_labels = np.eye(10, dtype='float32')[y_test]

        # Concatenate train and test labels along the first axis to get the final labels array
        data_sets.labels = np.concatenate((train_labels, test_labels), axis=0)
        #data_sets.labels = np.eye(10, dtype='float32')[np.concatenate((y_train, y_test), axis=0)]
        #data_sets.labels = np.concatenate((y_train, y_test), axis=0)
        print("Hello i am printing fr:")
        print(data_sets.labels.shape)
        print(data_sets.data.shape)
    # If we want to assign random labels to the data
    if random_labels:
        labels = np.zeros(data_sets.labels.shape)
        labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
        labels[np.arange(len(labels)), labels_index] = 1
        data_sets.labels = labels
    return data_sets

def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = int(perc(percent_of_train, data_sets_org.data.shape[0]))
    start_test_index = int(stop_train_index)
    if percent_of_train > min_test_data:
        start_test_index = int(perc(min_test_data, data_sets_org.data.shape[0]))
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets

# This function was used for dtype conversion, which might not be necessary in the simplified context
# However, if needed, TF2 supports these types directly without conversion



# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# import scipy.io as sio
# import os
# import sys
# import tensorflow as tf


# def load_data(name, random_labels=False):
# 	"""Load the data
# 	name - the name of the dataset
# 	random_labels - True if we want to return random labels to the dataset
# 	return object with data and labels"""
# 	print ('Loading Data...')
# 	C = type('type_C', (object,), {})
# 	data_sets = C()
# 	if name.split('/')[-1] == 'MNIST':
# 		data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0]) + "/data/MNIST_data/", one_hot=True)
# 		data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis=0)
# 		data_sets.labels = np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis=0)
# 	else:
# 		d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
# 		F = d['F']
# 		y = d['y']
# 		C = type('type_C', (object,), {})
# 		data_sets = C()
# 		data_sets.data = F
# 		data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
# 	# If we want to assign random labels to the  data
# 	if random_labels:
# 		labels = np.zeros(data_sets.labels.shape)
# 		labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
# 		labels[np.arange(len(labels)), labels_index] = 1
# 		data_sets.labels = labels
# 	return data_sets


# def shuffle_in_unison_inplace(a, b):
# 	"""Shuffle the arrays randomly"""
# 	assert len(a) == len(b)
# 	p = np.random.permutation(len(a))
# 	return a[p], b[p]


# def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
# 	"""Divided the data to train and test and shuffle it"""
# 	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
# 	C = type('type_C', (object,), {})
# 	data_sets = C()
# 	stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
# 	start_test_index = stop_train_index
# 	if percent_of_train > min_test_data:
# 		start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
# 	data_sets.train = C()
# 	data_sets.test = C()
# 	if shuffle_data:
# 		shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
# 	else:
# 		shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
# 	data_sets.train.data = shuffled_data[:stop_train_index, :]
# 	data_sets.train.labels = shuffled_labels[:stop_train_index, :]
# 	data_sets.test.data = shuffled_data[start_test_index:, :]
# 	data_sets.test.labels = shuffled_labels[start_test_index:, :]
# 	return data_sets


def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)
