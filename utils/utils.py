import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os 

import PIL

def preprocess_images(images, dims=(-1, 28, 28, 1)):
    images = images.reshape(dims) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def compute_output_dims(input_dims, kernel_size, strides):
    output_shape = np.round((input_dims - kernel_size) / strides + 0.5) + 1.
    return np.array(output_shape, dtype=np.int32)


def generate_and_save_images(model, path, epoch, test_x, show_images=False):
  mean, logvar = model.encode(test_x)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)

  num_images = predictions.shape[0]
  grid_size = int(np.sqrt(num_images))

  fig = plt.figure(figsize=(grid_size, grid_size))

  for i in range(num_images):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(os.path.join(path, f'{model.prefix}_epoch_{epoch:04d}.png'))
  if show_images:
    plt.show()
  else:
    plt.close()
  

def animated_image_generation(model, path):
  anim_file = os.path.join(path, f'{model.prefix}.gif')

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(f'{path}/{model.prefix}_epoch*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


def plot_latent_images(model, path, n, digit_size=28, show_images=False):
  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size * n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.savefig(os.path.join(path, f'{model.prefix}_latent_embedding.png'))
  if show_images:
    plt.show()
  else:
    plt.close()
