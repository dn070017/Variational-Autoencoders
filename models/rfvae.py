import tensorflow as tf
from utils.losses import rfvae_loss
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
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), lambda_min=0.1, lambda_max=10, eta_s=6.4, eta_h=6.4, prefix='tcvae'):
    super(RFVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.loss_fn = rfvae_loss
    self.eta_s = eta_s
    self.eta_h = eta_h
    self.relevance = RelevanceLayer(latent_dim, lambda_min=lambda_min, lambda_max=lambda_max)

  def train_step_discriminator(self, batch, optimizers):
    self.set_discriminator_trainable(True)    
    with tf.GradientTape() as tape:
      mean, logvar = self.encode(batch)
      z = self.reparameterize(mean, logvar) * self.relevance.relevance_coefficient()
      z_perm = RFVAE.permute_dims(z)
      
      density = self.discriminator(tf.concat([z, z_perm], axis=0))
      labels = RFVAE.create_discriminator_label(z)

      discriminator_loss = tf.keras.losses.CategoricalCrossentropy()(labels, density)
      gradients = tape.gradient(discriminator_loss, self.trainable_variables)
      optimizers['discriminator'].apply_gradients(zip(gradients, self.trainable_variables))
    
    return discriminator_loss

  def set_discriminator_trainable(self, trainable=True):
    self.discriminator.trainable = trainable
    self.encoder.trainable = not trainable
    self.decoder.trainable = not trainable
    self.relevance.trainable = not trainable

  def relevance_score(self, **kwargs):
    return self.relevance.relevance_coefficient()[0]