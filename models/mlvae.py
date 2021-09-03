import numpy as np
import tensorflow as tf

from models.vae import BaseVAE
from utils.losses import mlvae_loss, compute_kl_divergence
from utils.utils import compute_output_dims

class MLVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), latent_group_dim=2, num_classes=10, prefix='tcvae'):
    super(MLVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.latent_group_dim = latent_group_dim
    self.loss_fn = mlvae_loss
    self.group_encoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape=self.input_dims),
        tf.keras.layers.Conv2D(
          filters=32, kernel_size=self.kernel_size,
          strides=self.strides, activation='relu'
        ),
        tf.keras.layers.Conv2D(
          filters=64, kernel_size=self.kernel_size,
          strides=self.strides, activation='relu'
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * self.latent_group_dim),
      ]
    )

    output_dims = compute_output_dims(
      input_dims=self.input_dims[:-1],
      kernel_size=self.kernel_size,
      strides=self.strides)
    output_dims = compute_output_dims(
      input_dims=output_dims,
      kernel_size=self.kernel_size,
      strides=self.strides)

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.latent_group_dim,)),
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

  def forward(self, batch, apply_sigmoid=False):
    mean, logvar = self.encode(batch)
    mean, logvar, unique_mean, unique_logvar = self.accumulate_group_evidence(mean, logvar, batch)
    z = self.reparameterize(mean, logvar)
    x_pred = self.decode(z, apply_sigmoid=apply_sigmoid)
  
    return mean, logvar, z, x_pred

  def encode(self, batch):
    group_mean, group_logvar = tf.split(self.group_encoder(batch['x']), num_or_size_splits=2, axis=1)
    style_mean, style_logvar = tf.split(self.encoder(batch['x']), num_or_size_splits=2, axis=1)
    mean = tf.concat([group_mean, style_mean], axis=1)
    logvar = tf.concat([group_logvar, style_logvar], axis=1)
    return mean, logvar

  def accumulate_group_evidence(self, mean, logvar, batch):
    group_mean = mean[:, 0:self.latent_group_dim]
    group_logvar = logvar[:, 0:self.latent_group_dim]

    acc_mean_unique = []
    acc_logvar_unique = []
    for i in tf.range(batch['y'].shape[1]):
      Xg_cond = batch['y'][:, i] == 1.0
      Xg_indices = tf.where(Xg_cond)
      if Xg_indices.shape[0] == 0:
        continue
      group_batch_mean = tf.squeeze(tf.gather(group_mean[:, 0:self.latent_group_dim], Xg_indices, axis=0))
      group_batch_logvar = tf.squeeze(tf.gather(group_logvar[:, 0:self.latent_group_dim], Xg_indices, axis=0))
      
      group_batch_mean = tf.reshape(group_batch_mean, (-1, self.latent_group_dim))
      group_batch_logvar = tf.reshape(group_batch_logvar, (-1, self.latent_group_dim))
      
      group_batch_var = tf.exp(group_batch_logvar) + 1e-7

      acc_group_batch_var = 1 / tf.reduce_sum(1 / group_batch_var, axis=0)
      acc_group_batch_mean = tf.expand_dims(tf.reduce_sum(group_batch_mean * 1 / group_batch_var, axis=0) * acc_group_batch_var, 0)
      acc_group_batch_logvar = tf.expand_dims(tf.math.log(acc_group_batch_var), 0)
      group_mean = tf.where(tf.expand_dims(Xg_cond, axis=1), acc_group_batch_mean, group_mean)
      group_logvar = tf.where(tf.expand_dims(Xg_cond, axis=1), acc_group_batch_logvar, group_logvar)

      acc_mean_unique.append(acc_group_batch_mean)
      acc_logvar_unique.append(acc_group_batch_logvar)

    acc_mean = tf.concat([group_mean, mean[:, self.latent_group_dim:]], axis=1)
    acc_logvar =  tf.concat([group_logvar, logvar[:, self.latent_group_dim:]], axis=1)
    
    return acc_mean, acc_logvar, tf.concat(acc_mean_unique, axis=0), tf.concat(acc_logvar_unique, axis=0)

  def average_kl_divergence(self, batch):
    mean, logvar = self.encode(batch)
    mean, logvar, unique_mean, unique_logvar = self.accumulate_group_evidence(mean, logvar, batch)

    group_kl_divergence = compute_kl_divergence(unique_mean, unique_logvar)
    #print(group_kl_divergence)
    group_kl_divergence = tf.reduce_mean(group_kl_divergence, axis=0)

    kl_divergence = tf.reduce_mean(compute_kl_divergence(mean, logvar), axis=0)

    return tf.squeeze(tf.concat([group_kl_divergence, kl_divergence[self.latent_group_dim:]], axis=-1))