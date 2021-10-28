
import numpy as np
import tensorflow as tf

def data_generator(X, y):
  for image, label in zip(X, y):
    yield {'x': image, 'y': label}

def preprocess_images(images, dims=(-1, 28, 28, 1)):
    images = images.reshape(dims) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def get_dataset(task, train_size, test_size, batch_size):
  if task.lower() == 'mnist':
    num_classes = 10
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    train_labels = tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

  train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types={'x': tf.float32, 'y': tf.float32},
    args=(train_images, train_labels)
  ).shuffle(train_size).take(train_size).batch(batch_size)

  test_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types={'x': tf.float32, 'y': tf.float32},
    args=(test_images, test_labels)
  ).shuffle(test_size).take(test_size).batch(test_size)

  return train_dataset, test_dataset 

def get_dataset_params(task):
    if task.lower() == 'mnist':
      input_dims = (28, 28, 1)
      kernel_size = (3, 3)
      strides = (2, 2)
    
    return input_dims, kernel_size, strides