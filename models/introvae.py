import numpy as np
import tensorflow as tf

from utils.losses import compute_kl_divergence, compute_log_bernouli_pdf
from models.betavae import BetaVAE

class IntroVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(IntroVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)

  def train_step(self, batch, optimizers, beta=1.0):
    self.set_generator_trainable(False)
    with tf.GradientTape() as tape:
      encoder_loss, reconstructed_loss, kl_divergence, z, z_f = self.encoder_loss_fn(batch, beta=beta)
      
      gradients = tape.gradient(encoder_loss, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))

    self.set_generator_trainable(True)
    with tf.GradientTape() as tape:
      decoder_loss = self.decoder_loss_fn(batch, z, z_f, beta=beta)
      gradients = tape.gradient(decoder_loss, self.trainable_variables)
      optimizers['secondary'].apply_gradients(zip(gradients, self.trainable_variables))
        
    return encoder_loss + decoder_loss, reconstructed_loss, kl_divergence

  def set_generator_trainable(self, trainable=True):
    self.encoder.trainable = not trainable
    self.decoder.trainable = trainable

  def encoder_loss_fn(self, batch, beta=1.0):
    # implementation of Algorithm 1 from Daniel and Tamar 2021 (not specified additionally)
    # some part is from Algorithm 1 from Huang et al 2018 (specified as IntroVAE)
    # line 2
    s = tf.cast(1 / tf.reduce_prod(batch['x'].shape[1:]), dtype=tf.float32)

    # line 4, 5, 8
    mean_z, logvar_z, z, x_r = self.forward(batch)
    x_r_norm = tf.stop_gradient(tf.keras.activations.sigmoid(x_r))

    # line 6
    z_f = tf.random.normal(shape=z.shape)
    
    # line 8
    x_f_norm = tf.stop_gradient(self.decode(z_f, apply_sigmoid=True))

    # line 11
    logpx = compute_log_bernouli_pdf(x_r, batch['x'])
    logpx = tf.reduce_sum(logpx, axis=[1, 2, 3])
    kl_divergence_x = tf.reduce_sum(compute_kl_divergence(mean_z, logvar_z), axis=1)
    elbo_x = s * tf.reduce_mean(logpx - beta * kl_divergence_x)

    # line 9, line 10
    mean_z_rr, logvar_z_rr, z_rr, x_rr = self.forward({'x': x_r_norm}) # added
    mean_z_ff, logvar_z_ff, z_ff, x_ff = self.forward({'x': x_f_norm})

    # line 12, 13 
    # line 8, 9 from IntroVAE
    logpx_rr = compute_log_bernouli_pdf(x_rr, x_r_norm)
    logpx_rr = tf.reduce_sum(logpx_rr, axis=[1, 2, 3])
    kl_divergence_r = tf.reduce_sum(compute_kl_divergence(mean_z_rr, logvar_z_rr), axis=1)
    elbo_r = logpx_rr - beta * kl_divergence_r

    #print(tf.reduce_mean(tf.math.exp(2 * s * logpx_rr)).numpy(), tf.reduce_mean(2 * s * logpx_rr).numpy())

    # line 12, 13
    logpx_ff = compute_log_bernouli_pdf(x_ff, x_f_norm)
    logpx_ff = tf.reduce_sum(logpx_ff, axis=[1, 2, 3])
    kl_divergence_f = tf.reduce_sum(compute_kl_divergence(mean_z_ff, logvar_z_ff), axis=1)
    elbo_f = logpx_ff - beta * kl_divergence_f
    exp_elbo_f = 1 / 2 * (tf.reduce_mean(tf.math.exp(2 * s * elbo_r)) + tf.reduce_mean(tf.math.exp(2 * s * elbo_f)))

    # line 13, line 14 in Algorithm 1
    loss_encoder = -(elbo_x - exp_elbo_f)
    
    #print(elbo_x.numpy(), tf.reduce_mean(tf.math.exp(2 * s * elbo_r)).numpy(), tf.reduce_mean(tf.math.exp(2 * s * elbo_f)).numpy())
    #print('ENC', loss_encoder.numpy())
    return loss_encoder, tf.reduce_mean(logpx), tf.reduce_mean(kl_divergence_x), z, z_f

  def decoder_loss_fn(self, batch, z, z_f, beta=1.0):
    # implementation from Daniel and Tamar 2021
    # line 2
    s = tf.cast(1 / tf.reduce_prod(batch['x'].shape[1:]), dtype=tf.float32)

    # line 18
    x_r = self.decode(z)
    x_f_norm = tf.stop_gradient(self.decode(z_f, apply_sigmoid=True))
    x_r_norm = tf.stop_gradient(tf.keras.activations.sigmoid(x_r))
    
    # line 21 (assume beta_rec = 1)
    logpx = compute_log_bernouli_pdf(x_r, batch['x'])
    logpx = tf.reduce_sum(logpx, axis=[1, 2, 3])
    elbo_x = tf.reduce_mean(logpx)

    # line 19
    mean_z_rr, logvar_z_rr, z_rr, x_rr = self.forward({'x': x_r_norm})
    mean_z_ff, logvar_z_ff, z_ff, x_ff = self.forward({'x': x_f_norm})

    # line 20
    #x_r = tf.stop_gradient(tf.keras.activations.sigmoid(x_r))
    #x_f = tf.stop_gradient(x_f)

    # line 12 from IntroVAE
    logpx_rr = compute_log_bernouli_pdf(x_rr, x_r_norm)
    logpx_rr = tf.reduce_sum(logpx_rr, axis=[1, 2, 3])
    kl_divergence_r = tf.reduce_sum(compute_kl_divergence(mean_z_rr, logvar_z_rr), axis=1)
    elbo_r = tf.reduce_mean(1e-8 * logpx_rr - beta * kl_divergence_r)

    # line 22
    logpx_ff = compute_log_bernouli_pdf(x_ff, x_f_norm)
    logpx_ff = tf.reduce_sum(logpx_ff, axis=[1, 2, 3])
    kl_divergence_f = tf.reduce_sum(compute_kl_divergence(mean_z_ff, logvar_z_ff), axis=1)
    elbo_f = tf.reduce_mean(1e-8 * logpx_ff - beta * kl_divergence_f)
    
    # line 23
    loss_decoder = -s * (elbo_x + elbo_r + elbo_f)

    #print('DEC', loss_decoder.numpy())
    return loss_decoder