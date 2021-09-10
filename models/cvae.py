import numpy as np
import tensorflow as tf

from models.betavae import BetaVAE

class CVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_classes=10, prefix='tcvae'):
    super(CVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.num_classes = num_classes

    self.cond_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(2 * self.latent_dim),
    ])

  def forward(self, batch, apply_sigmoid=False):
    mean_z, logvar_z = self.encode(batch)
    z_sample = self.reparameterize(mean_z, logvar_z)
    x_pred = self.decode({'z': z_sample, 'y': batch['y']}, apply_sigmoid=apply_sigmoid)

    return mean_z, logvar_z, z_sample, x_pred

  def encode(self, batch):
    params_z = self.encoder(batch['x'])
    mean_z_u, logvar_z_u = tf.split(
      self.cond_encoder(tf.concat([params_z, batch['y']], axis=1)),
      num_or_size_splits=2, axis=-1
    )
    return mean_z_u, logvar_z_u

  def decode(self, batch, apply_sigmoid=False):
    logits = self.decoder(tf.concat([batch['z'], batch['y']], axis=1))
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

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
      cond = kwargs['y'][0:num_samples]
    
    return self.decode({'z': eps, 'y': cond}, apply_sigmoid=True)