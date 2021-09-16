import tensorflow as tf
from models.betavae import BetaVAE
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior

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

class FactorVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='tcvae'):
    super(FactorVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.discriminator = Discriminator()
  
  def train_step(self, batch, optimizers, beta=1.0):
    self.set_discriminator_trainable(False)
    batch_0, batch_1 = tf.split(batch['x'], 2, axis=0) 
    split_batch = [{'x': batch_0}, {'x': batch_1}]

    with tf.GradientTape() as tape:
      elbo, logpx_z, kl_divergence = self.elbo(split_batch[0], beta)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))

    self.train_step_discriminator(split_batch, optimizers)
    return elbo, logpx_z, kl_divergence

  def train_step_discriminator(self, split_batch, optimizers):
    self.set_discriminator_trainable(True)    
    with tf.GradientTape() as tape:
      num_samples = split_batch[0]['x'].shape[0]
      mean_z0, logvar_z0 = self.encode(split_batch[0])
      z0_sample = self.reparameterize(mean_z0, logvar_z0)

      mean_z1, logvar_z1 = self.encode(split_batch[1])
      z1_sample = self.reparameterize(mean_z1, logvar_z1)
      z1_sample_perm = FactorVAE.permute_dims(z1_sample)

      density = self.discriminator(tf.concat([z0_sample, z1_sample_perm], axis=0))
      labels = FactorVAE.create_discriminator_label(num_samples)

      discriminator_loss = tf.keras.losses.CategoricalCrossentropy()(labels, density)
      gradients = tape.gradient(discriminator_loss, self.trainable_variables)
      optimizers['secondary'].apply_gradients(zip(gradients, self.trainable_variables))
    
    return discriminator_loss

  def set_discriminator_trainable(self, trainable=True):
    self.discriminator.trainable = trainable
    self.encoder.trainable = not trainable
    self.decoder.trainable = not trainable

  def elbo(self, batch, beta=1.0):
    mean_z, logvar_z, z_sample, x_pred = self.forward(batch)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])

    kl_divergence = tf.reduce_sum(compute_kl_divergence_standard_prior(mean_z, logvar_z), axis=1)

    density = self.discriminator(z_sample)
    tc_loss = tf.reduce_mean(density[:, 0] - density[:, 1])

    elbo = tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  @staticmethod
  def permute_dims(z):
    num_dims = z.shape[-1]
    z_perm = []
    for z_dim in tf.split(z, tf.ones(num_dims, dtype=tf.int32), axis=-1):
        # z_perm.append(tf.random.shuffle(z_dim)) # tf.random.shuffle not implemented for GPU
        z_perm.append(tf.gather(z_dim, tf.random.shuffle(tf.range(z_dim.shape[0])))) # hacky way to address the issue
 
    return tf.stop_gradient(tf.concat(z_perm, axis=1))

  @staticmethod
  def create_discriminator_label(num_samples):
    # [[1, 0],
    #  [1, 0],
    #  ...
    #  [0, 1],
    #  [0, 1]]
    # shape: (2n, 2)
    label_for_z = tf.concat([tf.ones((num_samples, 1)), tf.zeros((num_samples, 1))], axis=1)
    label_for_z_perm = tf.concat([tf.zeros((num_samples, 1)), tf.ones((num_samples, 1))], axis=1)
    return tf.concat([label_for_z, label_for_z_perm], axis=0)