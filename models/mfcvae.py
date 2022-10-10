import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.vlae import VLAE
from utils.utils import compute_output_dims
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior, compute_kl_divergence, compute_log_normal_pdf

class Parameters(tf.keras.layers.Layer):
  def __init__(self, latent_dim, num_components, num_facets):
    super().__init__()
    self.latent_dim = latent_dim
    self.num_components = num_components

    self.pi_y = []
    self.mean_z_y = []
    self.logvar_z_y = []
    for _ in range(num_facets):
      self.pi_y.append(self.add_weight('pi_y', (1, num_components)))
      # shape: (event, batch)
      self.mean_z_y.append(self.add_weight('mean_z_y', (latent_dim, num_components)))
      self.logvar_z_y.append(self.add_weight('logvar_z_y', (latent_dim, num_components)))

class MFCVAE(VLAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_components=10, num_facets=3, warmup=10, prefix='mfcvae'):
    super().__init__(latent_dim, input_dims, kernel_size, strides, num_facets, warmup, prefix)
    self.params = Parameters(latent_dim, num_components, num_facets)
    self.num_facets = num_facets

  def elbo(self, batch, **kwargs):
    training = kwargs['training'] if 'training' in kwargs else False
    beta = kwargs['beta'] if 'beta' in kwargs else 1.0

    if 'epoch' in kwargs:
      beta *= min(kwargs['epoch'] / self.warmup, 1.0)

    # We use y to denote c_j
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] eq (C.48) from Falck et al., 2021.
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
    hl_list, mean_list, var_list = self.encode(batch, training=training)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list, training=training)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=False)

    # term (a): logpx_z 
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])

    elbo = tf.reduce_mean(logpx_z)
    for j in range(self.num_facets):
      # transpose mean and logvar to (batch, event) and construct multivariate Gaussian
      dist_z_x = tfp.distributions.MultivariateNormalDiag(
        mean_list[j],
        tf.sqrt(var_list[j]) + 1e-7
      )
      dist_z_y = tfp.distributions.MultivariateNormalDiag(
        tf.transpose(self.params.mean_z_y[j], [1, 0]),
        tf.exp(0.5 * tf.transpose(self.params.logvar_z_y[j], [1, 0]))
      )
      dist_y = tfp.distributions.Categorical(logits=tf.squeeze(self.params.pi_y[j]))
      dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y)

      logpz_y = dist_z_y.log_prob(tf.expand_dims(z_sample_list[j], -2))
      logpy = tf.math.log(tf.keras.activations.softmax(self.params.pi_y[j]) + 1e-7)
      
      logpz = dist_z.log_prob(tf.expand_dims(z_sample_list[j], -2))

      # py_z = pz_y * py / pz (modified from original implementation)
      # logpy_z = logpz_y + logpy - logpz
      #logpy_z = logpz_y + logpy - logpz
      #py_z = tf.exp(logpy_z)
      py_z = tf.keras.activations.softmax(logpz_y + logpy)
      logpy_z = tf.math.log(py_z + 1e-7)

      # term (b): 𝚺_j𝚺_y[py_z(logpz_y)]
      py_z_logpz_y = py_z * logpz_y

      # term (c): 𝚺_j𝚺_y[py_z(logpy)]
      py_z_logpy = py_z * logpy

      # term (d): 𝚺_j[logqz_x]
      logqz_x = dist_z_x.log_prob(z_sample_list[j])
      
      # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
      py_z_logpy_z = py_z * logpy_z

      elbo += beta * tf.reduce_mean(tf.reduce_sum(py_z_logpz_y, axis=-1))
      elbo += beta * tf.reduce_mean(tf.reduce_sum(py_z_logpy, axis=-1))
      elbo += beta * tf.reduce_mean(tf.reduce_sum(-py_z_logpy_z, axis=-1))
      elbo += beta * tf.reduce_mean(-logqz_x)
    
    return tf.reduce_mean(elbo), tf.reduce_mean(logpx_z), tf.reduce_mean(logpx_z - elbo) / beta

  def qy_x(self, batch):
    hl_list, mean_list, var_list = self.encode(batch, training=False)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list, training=False)

    qy_x = []
    for j in range(self.num_facets):
      dist_z_y = tfp.distributions.MultivariateNormalDiag(
        tf.transpose(self.params.mean_z_y[j], [1, 0]),
        tf.exp(0.5 * tf.transpose(self.params.logvar_z_y[j], [1, 0]))
      )

      # reshape to be broadcastable (batch, batch, event)
      pz_y = dist_z_y.log_prob(tf.expand_dims(z_sample_list[j], -2))
      py = tf.math.log(tf.keras.activations.softmax(self.params.pi_y[j]) + 1e-7)
      qy_x.append(tf.keras.activations.softmax(pz_y + py))

    return qy_x

  def generate(self, z=None, facet=3, num_generated_images=15, **kwargs):
    if facet > self.num_facets:
      facet = self.num_facets

    training = False

    if z is None:
      z = tf.zeros(shape=(num_generated_images, self.latent_dim))

    num_generated_images = z.shape[0]

    if 'target' in kwargs:
      target = kwargs['target']
      temperature = kwargs['temperature'] if 'temperature' in kwargs else 0.8
      z = tf.random.normal(
        shape=(num_generated_images, self.latent_dim),
        mean=self.params.mean_z_y[facet - 1][:, target],
        stddev=temperature * tf.exp(0.5 * self.params.logvar_z_y[facet - 1][:, target]),
        dtype=tf.float32
      )
    
    z_hat_list = [None] * self.num_facets
    z_sample_list = [tf.zeros(shape=(num_generated_images, self.latent_dim))] * self.num_facets
    z_sample_list[facet - 1] = z

    index = self.num_facets - 1
    z = z_sample_list[index]
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z, training=training)], axis=1), training=training)
    
    for index in reversed(range(0, self.num_facets - 1)):
      z_hat_prior = z_hat_list[index + 1]
      z = z_sample_list[index]
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z, training=training)], axis=1), training=training)

    return self.decode(z_hat_list[0], apply_sigmoid=True)