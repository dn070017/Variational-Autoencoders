import numpy as np
import tensorflow as tf
from utils.utils import compute_output_dims
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior

class VLAE(tf.keras.Model):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_layers=3, warmup=10, prefix='vlae'):
    super().__init__()
    self.prefix = prefix
    self.latent_dim = latent_dim
    self.input_dims = np.array(input_dims, dtype=np.int32)
    self.kernel_size = np.array(kernel_size, dtype=np.int32)
    self.num_layers = num_layers
    self.strides = np.array(strides, dtype=np.int32)
    self.output_dims = None
    self.warmup = warmup

    self.inference_layers = []
    self.inference_mean_linear_layers = []
    self.inference_var_softplus_layers = []
    self.generative_layers_u = []
    self.generative_layers_v = []

    # x' (image) -> x (CNN transformation) -> z
    # p(z_0|x) is parameterized using a convolutional neural network (modified from original paper)
    self.encoder = tf.keras.Sequential([
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
    ])

    
    for i in range(0, self.num_layers):
      # eq (12) from Zhao et al., 2017
      self.inference_layers.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(128, activation='elu'),
          tf.keras.layers.BatchNormalization()
        ])
      )
      # eq (13) from Zhao et al., 2017
      self.inference_mean_linear_layers.append(
        tf.keras.layers.Dense(self.latent_dim)
      )
      self.inference_var_softplus_layers.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(self.latent_dim),
          tf.keras.layers.Lambda(lambda x: tf.math.softplus(x))
        ])
      )

      self.generative_layers_u.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(128, activation='elu'),
          tf.keras.layers.BatchNormalization()
        ])
      )
      self.generative_layers_v.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(128, activation='elu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(256, activation='elu'),
          tf.keras.layers.BatchNormalization(), 
        ])
      )

    self.output_dims = compute_output_dims(
      input_dims=self.input_dims[:-1],
      kernel_size=self.kernel_size,
      strides=self.strides
    )
    self.output_dims = compute_output_dims(
      input_dims=self.output_dims,
      kernel_size=self.kernel_size,
      strides=self.strides
    )
 
    # after reconstruct z_0 from the stochastic downward pass, transform z_0 back to x'
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(units=tf.reduce_prod(self.output_dims) * 32, activation='relu'),
      tf.keras.layers.Reshape(target_shape=(self.output_dims[0], self.output_dims[1], 32)),
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

  def elbo(self, batch, **kwargs):
    training = kwargs['training'] if 'training' in kwargs else False
    beta = kwargs['beta'] if 'beta' in kwargs else 1.0

    if 'epoch' in kwargs:
      beta *= min(kwargs['epoch'] / self.warmup, 1.0)

    hl_list, mean_list, var_list = self.encode(batch, training=training)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list, training=training)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=False)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    
    for i in range(self.num_layers):
      if i == 0:
        kl_divergence = compute_kl_divergence_standard_prior(mean_list[0], tf.math.log(var_list[0] + 1e-7))
        kl_divergence = tf.reduce_sum(kl_divergence, axis=1)
      else:
        kl_divergence += tf.reduce_sum(compute_kl_divergence_standard_prior(mean_list[i], tf.math.log(var_list[i] + 1e-7)), axis=1)

    elbo = tf.reduce_mean(logpx_z - beta * kl_divergence)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def train_step(self, batch, optimizers, **kwargs):
    with tf.GradientTape() as tape:
      kwargs['training'] = True
      elbo, logpx_z, kl_divergence = self.elbo(batch, **kwargs)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))
        
      return elbo, logpx_z, kl_divergence

  def forward(self, batch, apply_sigmoid=False, training=False):
    hl_list, mean_list, var_list = self.encode(batch, training=training)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list, training=training)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=apply_sigmoid)
  
    return tf.concat(mean_list, axis=1), tf.concat(var_list, axis=1), hl_list, x_pred

  def encode(self, batch, training=False):
    hl_list = []
    mean_list = []
    var_list = []
    hl_list.append(self.encoder(batch['x']))
    for l in range(1, self.num_layers + 1):
      # Eq (12) h_l = gl(h_l-1)
      hl = self.inference_layers[l-1](hl_list[l-1], training=training)
      hl_list.append(hl)
      # Eq (13) mean = mu_l(h_l), var = var_l(h_l)
      mean_list.append(self.inference_mean_linear_layers[l-1](hl))
      var_list.append(self.inference_var_softplus_layers[l-1](hl))
    
    # If num_layers = 2
    # hl_list: [x0, h1, h2]
    # mean_hat_list: [mu1, mu2]
    # var_hat_list: [var1, var2]

    return hl_list, mean_list, var_list

  def encode_across_layers(self, mean_list, var_list, training=False):
    z_sample_list = [None] * self.num_layers
    z_hat_list = [None] * self.num_layers
    
    # process the last layer
    index = self.num_layers - 1
    z = self.reparameterize(mean_list[index], var_list[index], training=training)
    z_sample_list[index] = z
    # eq (8)
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z, training=training)], axis=1), training=training)
    
    for index in reversed(range(0, self.num_layers - 1)):
      # z_i (input for layer i) is sampled from the distribution define in layer i + 1.
      # e.g. the prior for the generative distribution mean_p1, var_p1 for layer 1 takes the z sampled from layer 2 as the input.
      z_hat_prior = z_hat_list[index + 1]
      z = self.reparameterize(mean_list[index], var_list[index], training=training)
      z_sample_list[index] = z
      # eq (9)
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z, training=training)], axis=1), training=training)

    return z_sample_list, z_hat_list

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    
    return logits

  def generate(self, z=None, layer=3, num_generated_images=15, **kwargs):
    if layer > self.num_layers:
      layer = self.num_layers
    if z is None:
      z = tf.zeros(shape=(num_generated_images, self.latent_dim))
    
    num_generated_images = z.shape[0]

    z_hat_list = [None] * self.num_layers
    z_sample_list = [tf.zeros(shape=(num_generated_images, self.latent_dim))] * self.num_layers
    z_sample_list[layer - 1] = z

    index = self.num_layers - 1
    z = z_sample_list[index]
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z, training=False)], axis=1), training=False)
    
    for index in reversed(range(0, self.num_layers - 1)):
      z_hat_prior = z_hat_list[index + 1]
      z = z_sample_list[index]
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z, training=False)], axis=1), training=False)

    return self.decode(z_hat_list[0], apply_sigmoid=True)

  def reparameterize(self, mean, var, training=True):
    if not training:
      return mean
    eps = tf.random.normal(shape=mean.shape) # each distribution has its own epsilon
    std = tf.math.sqrt(var)
    return eps * std + mean

  def average_kl_divergence(self, batch, layer=3):
    if layer > self.num_layers:
      layer = self.num_layers
    hl_list, mean_list, var_list = self.encode(batch, training=False)

    return tf.reduce_mean(compute_kl_divergence_standard_prior(mean_list[layer-1], tf.math.log(var_list[layer-1] + 1e-7)), axis=0)