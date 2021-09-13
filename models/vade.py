import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.betavae import BetaVAE
from utils.losses import compute_log_normal_pdf, compute_log_bernouli_pdf

class Parameters(tf.keras.layers.Layer):
  def __init__(self, latent_dim, num_components):
    super().__init__()
    self.latent_dim = latent_dim
    self.num_components = num_components
    self.pi_y = self.add_weight('pi_y', (1, num_components))

    # shape: (event, batch)
    self.mu_z_comp = self.add_weight('mu_z_comp', (latent_dim, num_components))
    self.logvar_z_comp = self.add_weight('logvar_z_comp', (latent_dim, num_components))

class VaDE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_components=10, prefix='vade'):
    super().__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.num_components = num_components
    self.params = Parameters(latent_dim, num_components)

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    var = tf.keras.activations.softplus(logvar)
    std = tf.math.sqrt(var)
    return eps * std + mean

  def elbo(self, batch, beta=1.0):
    mean_z_x, logvar_z_x = self.encode(batch)
    z_sample = self.reparameterize(mean_z_x, logvar_z_x)
    x_pred = self.decode(z_sample, apply_sigmoid=False)

    #dist_z_y = tfp.distributions.Independent(
    #  tfp.distributions.Normal(
    #    tf.transpose(self.params.mu_z_comp, [1, 0]),
    #    tf.exp(0.5 * tf.transpose(self.params.logvar_z_comp, [1, 0]))
    #  ), reinterpreted_batch_ndims=1
    #)
    dist_z_y = tfp.distributions.MultivariateNormalDiag(
      tf.transpose(self.params.mu_z_comp, [1, 0]),
      tf.exp(0.5 * tf.transpose(self.params.logvar_z_comp, [1, 0]))
    )

    #print(tf.reduce_max(tf.abs(mean_z_x)).numpy())
    #print(tf.reduce_max(tf.abs(logvar_z_x)).numpy())
    #print(tf.reduce_max(tf.abs(mean_x_z)).numpy())
    #print(tf.reduce_max(tf.abs(logvar_x_z)).numpy())
    #print(self.params.pi_y)
    dist_y = tfp.distributions.Categorical(logits=tf.squeeze(self.params.pi_y))
    dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    logpz = dist_z.log_prob(z_sample)
    logqz_x = compute_log_normal_pdf(mean_z_x, logvar_z_x, z_sample)
    logqz_x = tf.reduce_sum(logqz_x, axis=1)

    elbo = logpx_z - (logqz_x - logpz)
    #print(tf.reduce_mean(elbo))

    return tf.reduce_mean(elbo), tf.reduce_mean(logpx_z), tf.reduce_mean(logqz_x - logpz)

  def qy_x(self, batch):
    mean_z_x, logvar_z_x = self.encode(batch)
    z_sample = self.reparameterize(mean_z_x, logvar_z_x)
    dist_z_y = tfp.distributions.MultivariateNormalDiag(
        tf.transpose(self.params.mu_z_comp, [1, 0]),
        tf.exp(0.5 * tf.transpose(self.params.logvar_z_comp, [1, 0]))
    )

    # reshape to be broadcastable (batch, batch, event)
    pz_y = dist_z_y.log_prob(tf.expand_dims(z_sample, -2))
    py = tf.math.log(tf.keras.activations.softmax(self.params.pi_y) + 1e-7)
    qy_x = tf.keras.activations.softmax(pz_y + py)

    return qy_x