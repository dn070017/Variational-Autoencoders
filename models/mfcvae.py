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
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_components=10, num_facets=3, prefix='vae'):
    super().__init__(latent_dim, input_dims, kernel_size, strides, num_facets, prefix)
    self.params = Parameters(latent_dim, num_components, num_facets)
    self.num_facets = num_facets

  def elbo(self, batch, training=False, **kwargs):
    
    # We use y to denote c_j
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] eq (C.48) from Falck et al., 2021.
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
    hl_list, mean_list, var_list = self.encode(batch)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=False)

    # term (a): logpx_z 
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])

    #print('START')
    elbo = tf.reduce_mean(logpx_z)
    for j in range(self.num_facets):
      # transpose mean and logvar to (batch, event) and construct multivariate Gaussian
      dist_z_x = tfp.distributions.MultivariateNormalDiag(
        mean_list[j],
        tf.sqrt(var_list[j])
      )
      dist_z_y = tfp.distributions.MultivariateNormalDiag(
        tf.transpose(self.params.mean_z_y[j], [1, 0]),
        tf.sqrt(tf.exp(0.5 * tf.transpose(self.params.logvar_z_y[j], [1, 0])))
      )
      dist_y = tfp.distributions.Categorical(logits=tf.squeeze(self.params.pi_y[j]))
      dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y)

      logpz_y = dist_z_y.log_prob(tf.expand_dims(z_sample_list[j], -2))#, -100, 1e7)
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
      #logqz_x = compute_log_normal_pdf(mean_q_list[j], tf.math.log(var_q_list[j] + 1e-7), z_list[j])#, -100, 1e7)
      logqz_x = dist_z_x.log_prob(z_sample_list[j])
      
      
      # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
      py_z_logpy_z = py_z * logpy_z

      elbo += tf.reduce_mean(tf.reduce_sum(py_z_logpz_y, axis=-1))
      elbo += tf.reduce_mean(tf.reduce_sum(py_z_logpy, axis=-1))
      elbo += tf.reduce_mean(tf.reduce_sum(-py_z_logpy_z, axis=-1))
      elbo += tf.reduce_mean(-logqz_x)#tf.reduce_mean(tf.reduce_sum(-logqz_x, axis=-1))
    
      #elbo += (tf.reduce_mean(logpz) - tf.reduce_mean(logqz_x))
      #elbo += (tf.reduce_sum(py_z_logpz_y + py_z_logpy - py_z_logpy_z, axis=-1) - tf.reduce_sum(logqz_x, axis=-1))

      """dist_z_y = tfp.distributions.MultivariateNormalDiag(
        tf.transpose(self.params.mean_z_y[j], [1, 0]),
        tf.sqrt(tf.exp(0.5 * tf.transpose(self.params.logvar_z_y[j], [1, 0]))) #sqrt
      )

      dist_y = tfp.distributions.Categorical(logits=tf.squeeze(self.params.pi_y[j]))
      dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y)

      logpz = dist_z.log_prob(z_list[j])
      logqz_x = compute_log_normal_pdf(mean_hat_ist[j], tf.math.log(var_hat_list[j] + 1e-7), z_list[j])
      logqz_x = tf.reduce_sum(logqz_x, axis=1)

      elbo += tf.reduce_mean(logpz - logqz_x)"""
      #print(tf.reduce_mean(logpz), tf.reduce_mean(logqz_x))
      #print('\t- ', elbo.numpy(), tf.reduce_mean(logpx_z).numpy(), tf.reduce_mean(tf.reduce_sum(py_z_logpz_y, axis=-1)).numpy(), tf.reduce_mean(tf.reduce_sum(py_z_logpy, axis=-1)).numpy(),  tf.reduce_mean(tf.reduce_sum(-py_z_logpy_z, axis=-1)).numpy(), tf.reduce_mean(tf.reduce_sum(-logqz_x, axis=-1)).numpy())
    #print(tf.reduce_max(tf.math.abs(logpx_z)).numpy(), tf.reduce_max(tf.math.abs(py_z_logpz_y)).numpy(), tf.reduce_max(tf.math.abs(py_z_logpy)).numpy(), tf.reduce_max(tf.math.abs(py_z_logpy_z)).numpy(), tf.reduce_max(tf.math.abs(logqz_x)).numpy())
    #print(tf.reduce_max(tf.math.abs(logpy_z)), tf.reduce_max(tf.math.abs(logpz_y)))
    return tf.reduce_mean(elbo), tf.reduce_mean(logpx_z), tf.reduce_mean(elbo - logpx_z)

  def generate(self, z=None, facet=3, num_generated_images=15, **kwargs):
    if z is None:
      z = tf.zeros(shape=(num_generated_images, self.latent_dim))

    num_generated_images = z.shape[0]

    if 'target' in kwargs:
        target = kwargs['target']
        z = tf.random.normal(
            shape=(num_generated_images, self.latent_dim),
            mean=self.params.mean_z_y[facet - 1][:, target],
            stddev=tf.sqrt(tf.exp(0.5 * self.params.logvar_z_y[facet - 1][:, target])),
            dtype=tf.float32
        )
    
    z_hat_list = [None] * self.num_layers
    z_sample_list = [tf.zeros(shape=(num_generated_images, self.latent_dim))] * self.num_layers
    z_sample_list[facet - 1] = z

    index = self.num_facets - 1
    z = z_sample_list[index]
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z)], axis=1))
    
    for index in reversed(range(0, self.num_facets - 1)):
      z_hat_prior = z_hat_list[index + 1]
      z = z_sample_list[index]
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z)], axis=1))

    return self.decode(z_hat_list[0], apply_sigmoid=True)