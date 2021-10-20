import glob
import imageio
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os 

import PIL

def compute_output_dims(input_dims, kernel_size, strides):
    output_shape = np.round((input_dims - kernel_size) / strides + 0.5) + 1.
    return np.array(output_shape, dtype=np.int32)

def generate_and_save_images(model, path, epoch, test_x, show_images=False):
  mean, logvar, z, x_pred = model.forward(test_x, apply_sigmoid=True)

  num_images = x_pred.shape[0]
  grid_size = int(np.sqrt(num_images))

  fig = plt.figure(figsize=(grid_size, grid_size))

  for i in range(num_images):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(x_pred[i, :, :, 0], cmap='gray')
    plt.axis('off')

  filename = os.path.join(path, f'{model.prefix}_epoch_{epoch:04d}.png')
  plt.savefig(filename)
  if show_images:
    plt.show()
  else:
    plt.close()
  
  mlflow.log_artifact(filename)

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
  
  mlflow.log_artifact(anim_file)


def plot_latent_images(model, batch, path, n, epoch, digit_size=28, show_images=False):
  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size * n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  kl_divergence = model.average_kl_divergence(batch=batch)

  importance_order = tf.argsort(
    kl_divergence, axis=-1, direction='DESCENDING', stable=False, name=None
  )

  latent_dim = model.latent_dim
  if type(model).__name__ == 'MLVAE':
    latent_dim += model.latent_content_dim
    importance_order = tf.argsort(
      kl_divergence[model.latent_content_dim:],
      axis=-1, direction='DESCENDING', stable=False, name=None
    )
    importance_order += model.latent_content_dim

  target = 0
  for i, yi in enumerate(grid_x):
    if target >= 10:
      target = 0
    for j, xi in enumerate(grid_y):
      z = np.zeros((1, latent_dim))
      # select the top 2 most relevant latent dimensions
      z[0, importance_order[0]] = xi
      z[0, importance_order[1]] = yi
      z = tf.convert_to_tensor(z, dtype=tf.float32)
      x_decoded = model.generate(z, target=target)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit.numpy()
    target += 1  

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')

  filename = os.path.join(path, f'{model.prefix}_latent_embedding_epoch_{epoch:04d}.png')
  plt.savefig(filename)
  if show_images:
    plt.show()
  else:
    plt.close()
  mlflow.log_artifact(filename)