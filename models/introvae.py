import numpy as np
import tensorflow as tf

from utils.losses import soft_introvae_encoder_loss, soft_introvae_decoder_loss
from models.vae import BaseVAE

class IntroVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(IntroVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.loss_fn_encoder = soft_introvae_encoder_loss
    self.loss_fn_decoder = soft_introvae_decoder_loss

  def train_step(self, batch, optimizers, beta=1.0):
    self.set_generator_trainable(False)
    with tf.GradientTape() as tape:
      encoder_loss, reconstructed_loss, kl_divergence = self.loss_fn_encoder(self, batch, beta=beta)
      
      gradients = tape.gradient(encoder_loss, self.trainable_variables)
      optimizers['vae'].apply_gradients(zip(gradients, self.trainable_variables))

    self.set_generator_trainable(True)
    with tf.GradientTape() as tape:
      decoder_loss = self.loss_fn_decoder(self, batch, beta=beta)
      gradients = tape.gradient(decoder_loss, self.trainable_variables)
      optimizers['discriminator'].apply_gradients(zip(gradients, self.trainable_variables))
        
    return encoder_loss + decoder_loss, reconstructed_loss, kl_divergence

  def set_generator_trainable(self, trainable=True):
    self.encoder.trainable = not trainable
    self.decoder.trainable = trainable