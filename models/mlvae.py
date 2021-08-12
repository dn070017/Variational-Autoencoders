import numpy as np
import tensorflow as tf

from models.vae import BaseVAE

class MLVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), latent_group_dim=2, prefix='tcvae'):
    super(MLVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.latent_group_dim = latent_group_dim 

  def encode(self, batch):
    mean, logvar = tf.split(self.encoder(batch['x']), num_or_size_splits=2, axis=1)
    mean, logvar = self.accumulate_group_evidence(mean, logvar, batch)
    return mean, logvar

  def accumulate_group_evidence(self, mean, logvar, batch):
    group_mean = mean[:, 0:self.latent_group_dim]
    group_logvar = logvar[:, 0:self.latent_group_dim]

    for i in tf.range(batch['y'].shape[1]):
      Xg_cond = batch['y'][:, i] == 1.0
      Xg_indices = tf.where(Xg_cond)
      group_batch_mean = tf.squeeze(tf.gather(group_mean[:, 0:self.latent_group_dim], Xg_indices, axis=0))
      group_batch_logvar = tf.squeeze(tf.gather(group_logvar[:, 0:self.latent_group_dim], Xg_indices, axis=0))
      group_batch_var = tf.exp(group_batch_logvar)

      acc_group_batch_var = 1 / tf.reduce_sum(1 / group_batch_var, axis=0)
      acc_group_batch_mean = tf.expand_dims(tf.reduce_sum(group_batch_mean * 1 / group_batch_var, axis=0) * acc_group_batch_var, 0)
      acc_group_batch_logvar = tf.expand_dims(tf.math.log(acc_group_batch_var), 0)
      group_mean = tf.where(tf.expand_dims(Xg_cond, axis=1), acc_group_batch_mean, group_mean)
      group_logvar = tf.where(tf.expand_dims(Xg_cond, axis=1), acc_group_batch_logvar, group_logvar)

    acc_mean = tf.concat([group_mean, mean[:, self.latent_group_dim:]], axis=1)
    acc_logvar =  tf.concat([group_logvar, logvar[:, self.latent_group_dim:]], axis=1)
    
    return acc_mean, acc_logvar
