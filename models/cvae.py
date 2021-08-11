import numpy as np
import tensorflow as tf

from models.vae import BaseVAE
from utils.utils import compute_output_dims
from utils.losses import cvae_loss

class CVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_classes=10, prefix='tcvae'):
    super(CVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.loss_fn = cvae_loss
    self.num_classes = num_classes

    output_dims = compute_output_dims(
      input_dims=self.input_dims[:-1],
      kernel_size=self.kernel_size,
      strides=self.strides)
    output_dims = compute_output_dims(
      input_dims=output_dims,
      kernel_size=self.kernel_size,
      strides=self.strides)

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.num_classes,)),
      tf.keras.layers.Dense(units=tf.reduce_prod(output_dims) * 32, activation='relu'),
      tf.keras.layers.Reshape(target_shape=(output_dims[0], output_dims[1], 32)),
      tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=self.kernel_size, strides=2, 
        padding='same', activation='relu'),
      tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=self.kernel_size, strides=2,
        padding='same', activation='relu'),
      tf.keras.layers.Conv2DTranspose(
        filters=self.input_dims[-1], kernel_size=self.kernel_size, strides=1,
        padding='same')
    ])

    self.cond_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(2 * self.latent_dim),
    ])

  def forward(self, batch, apply_sigmoid=False):
    mean, logvar = self.encode(batch)
    z = self.reparameterize(mean, logvar)
    x_pred = self.decode({'z': z, 'y': batch['y']}, apply_sigmoid=apply_sigmoid)

    return mean, logvar, z, x_pred

  @tf.function
  def generate(self, eps=None, num_generated_images=15, **kwargs):
    if eps is None:
      eps = tf.random.normal(shape=(num_generated_images, self.latent_dim), dtype=tf.float32)

    num_samples = eps.shape[0]

    if 'y' not in kwargs and 'target' not in kwargs:
      cond = np.zeros((num_samples, self.num_classes))
      target = 0
      for i in range(num_samples):
        cond[i, target] = 1.0
        target += 1
        if target >= self.num_classes:
            target = 0
      cond = tf.convert_to_tensor(cond, dtype=tf.float32)
    elif 'target' in kwargs:
      cond = np.zeros((num_samples, self.num_classes))
      target = kwargs['target']
      for i in range(num_samples):
        cond[i, target] = 1.0
    else:
      cond = kwargs['y'][0: num_samples]
    
    return self.decode({'z': eps, 'y': cond}, apply_sigmoid=True)

  def encode(self, batch):
    z = self.encoder(batch['x'])
    mean, logvar = tf.split(self.cond_encoder(tf.concat([z, batch['y']], axis=1)), num_or_size_splits=2, axis=1)
    return mean, logvar

  def decode(self, batch, apply_sigmoid=False):
    logits = self.decoder(tf.concat([batch['z'], batch['y']], axis=1))
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits