import numpy as np
import tensorflow as tf
from models.vae import BaseVAE
from utils.utils import compute_output_dims
from utils.losses import tcvae_loss

class TCVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(TCVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.loss_fn = tcvae_loss
  
  def train_step(self, x_true, optimizer, beta=1.0):
    with tf.GradientTape() as tape:
      total_loss, reconstructed_loss, kl_divergence = self.loss_fn(self, x_true, beta)
      gradients = tape.gradient(total_loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
      return total_loss, reconstructed_loss, kl_divergence

