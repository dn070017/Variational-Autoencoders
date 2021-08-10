import numpy as np
import tensorflow as tf
from utils.utils import compute_output_dims
from utils.losses import vae_loss
class BaseVAE(tf.keras.Model):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='vae'):
    super(BaseVAE, self).__init__()
    self.prefix = prefix
    self.latent_dim = np.array(latent_dim, dtype=np.int32)
    self.input_dims = np.array(input_dims, dtype=np.int32)
    self.kernel_size = np.array(kernel_size, dtype=np.int32)
    self.strides = np.array(strides, dtype=np.int32)
    self.loss_fn = vae_loss

    self.encoder = tf.keras.Sequential(
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
        tf.keras.layers.Dense(2 * self.latent_dim),
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
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
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

  def train_step(self, x_true, optimizer, beta=1.0):
    with tf.GradientTape() as tape:
      total_loss, reconstructed_loss, kl_divergence = self.loss_fn(self, x_true, beta)
      gradients = tape.gradient(total_loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
      return total_loss, reconstructed_loss, kl_divergence

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits