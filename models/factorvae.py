import numpy as np
import tensorflow as tf
from models.vae import BaseVAE
from utils.losses import factorvae_loss

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(100),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(100),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    
    def call(self, z):
        return self.network(z)

class FactorVAE(BaseVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(FactorVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.loss_fn = factorvae_loss
    self.discriminator = Discriminator()
  
  def train_step(self, x_true, optimizer_vae, optimizer_d, beta=1.0):
    self.set_discriminator_trainable(False)
    with tf.GradientTape() as tape:
      total_loss, reconstructed_loss, kl_divergence = self.loss_fn(self, x_true, beta)
      gradients = tape.gradient(total_loss, self.trainable_variables)
      optimizer_vae.apply_gradients(zip(gradients, self.trainable_variables))

    self.set_discriminator_trainable(True)    
    with tf.GradientTape() as tape:
      mean, logvar = self.encode(x_true)
      z = self.reparameterize(mean, logvar)
      z_perm = FactorVAE.permute_dims(z)

      density = self.discriminator(tf.concat([z, z_perm], axis=0))
      labels = FactorVAE.create_discriminator_label(z)

      discriminator_loss = tf.keras.losses.CategoricalCrossentropy()(labels, density)
      gradients = tape.gradient(discriminator_loss, self.trainable_variables)
      optimizer_d.apply_gradients(zip(gradients, self.trainable_variables))
    
    return total_loss, reconstructed_loss, kl_divergence

  def set_discriminator_trainable(self, trainable=True):
    self.discriminator.trainable = trainable
    self.encoder.trainable = not trainable
    self.decoder.trainable = not trainable

  @staticmethod
  def permute_dims(z):
    num_dims = z.shape[1]
    z_perm = []
    for z_dim in tf.split(z, tf.ones(num_dims, dtype=tf.int32), axis=1):
        # z_perm.append(tf.random.shuffle(z_dim)) # tf.random.shuffle not implemented for GPU
        z_perm.append(tf.gather(z_dim, tf.random.shuffle(tf.range(z_dim.shape[0])))) # Hacky way to solve the issue
 
    return tf.concat(z_perm, axis=1)

  @staticmethod
  def create_discriminator_label(z):
    num_samples = z.shape[0]
    label_for_z = tf.concat([tf.ones((num_samples, 1)), tf.zeros((num_samples, 1))], axis=1)
    label_for_z_perm = tf.concat([tf.zeros((num_samples, 1)), tf.ones((num_samples, 1))], axis=1)
    return tf.concat([label_for_z, label_for_z_perm], axis=0)