import tensorflow as tf
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior
from models.factorvae import FactorVAE

class RelevanceLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, lambda_min=0.1, lambda_max=10.0):
        super().__init__()
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.relevance = self.add_weight(
            'relevance', (1, latent_dim),
            initializer=tf.keras.initializers.Constant(0.5) # sigmoid(5.) = 0.9933 / sigmoid(0.) = 0.5
        ) 

    def relevance_coefficient(self):
        rc = tf.keras.activations.relu(self.relevance)
        rc = tf.where(rc > 1., 1., rc)
        return rc

    def penalty_coefficient(self):
        rc = self.relevance_coefficient()
        return self.lambda_max + (self.lambda_min - self.lambda_max) * rc

class RFVAE(FactorVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), lambda_min=0.1, lambda_max=10, prefix='tcvae'):
    super(RFVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.relevance = RelevanceLayer(latent_dim, lambda_min=lambda_min, lambda_max=lambda_max)

  def train_step_discriminator(self, split_batch, optimizers):
    self.set_discriminator_trainable(True)    
    with tf.GradientTape() as tape:
      num_samples = split_batch[0]['x'].shape[0]
      mean_z0, logvar_z0 = self.encode(split_batch[0])
      z0_sample = self.reparameterize(mean_z0, logvar_z0)
      z0_sample *= self.relevance.relevance_coefficient()

      mean_z1, logvar_z1 = self.encode(split_batch[1])
      z1_sample = self.reparameterize(mean_z1, logvar_z1)
      z1_sample_perm = RFVAE.permute_dims(z1_sample)
      
      density = self.discriminator(tf.concat([z0_sample, z1_sample_perm], axis=0))
      labels = RFVAE.create_discriminator_label(num_samples)

      discriminator_loss = tf.keras.losses.CategoricalCrossentropy()(labels, density)
      gradients = tape.gradient(discriminator_loss, self.trainable_variables)
      optimizers['secondary'].apply_gradients(zip(gradients, self.trainable_variables))
    
    return discriminator_loss

  def elbo(self, batch, beta=1.0, eta_s=6.4, eta_h=6.4):
    rc = self.relevance.relevance_coefficient()       # R
    rc_penalty = self.relevance.penalty_coefficient() # lambda

    fractional_loss = -1 * tf.reduce_sum(rc * tf.math.log(rc + 1e-7) + (1 - rc) * tf.math.log((1 - rc) + 1e-7))

    mean_z, logvar_z, z_sample, x_pred = self.forward(batch)

    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])

    kl_divergence_not_reduced = compute_kl_divergence_standard_prior(mean_z, logvar_z)
    kl_divergence = tf.reduce_sum(kl_divergence_not_reduced, axis=-1)
    weighted_kl_divergence = tf.reduce_sum(rc_penalty * kl_divergence_not_reduced, axis=-1)

    density = self.discriminator(rc * z_sample)
    tc_loss = tf.reduce_mean(density[:, 0] - density[:, 1])

    elbo = tf.reduce_mean(logpx_z - (weighted_kl_divergence + (beta - 1) * tc_loss)) - eta_s * tf.reduce_sum(tf.abs(rc)) - eta_h * fractional_loss

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def set_discriminator_trainable(self, trainable=True):
    self.discriminator.trainable = trainable
    self.encoder.trainable = not trainable
    self.decoder.trainable = not trainable
    self.relevance.trainable = not trainable

  def relevance_score(self, **kwargs):
    return self.relevance.relevance_coefficient()[0]