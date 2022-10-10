import tensorflow as tf

from models.betavae import BetaVAE
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior, compute_total_correlation

class TCVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(TCVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
  
  def tcvae_loss(self, batch, beta=1.0):
    mean_z, logvar_z, z_sample, x_pred = self.forward(batch)

    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
  
    kl_divergence = tf.reduce_sum(compute_kl_divergence_standard_prior(mean_z, logvar_z), axis=1)
    tc_loss = compute_total_correlation(mean_z, logvar_z, z_sample)

    elbo = tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))
  
    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)