import numpy as np
import tensorflow as tf
from utils.utils import compute_output_dims
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior

class VLAE(tf.keras.Model):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_layers=3, prefix='vae'):
    super().__init__()
    self.prefix = prefix
    #self.latent_dim = (latent_dim // num_ladders) * num_ladders
    #self.latent_dim = np.array(latent_dim // num_ladders, dtype=np.int32)
    self.latent_dim = latent_dim
    self.input_dims = np.array(input_dims, dtype=np.int32)
    self.kernel_size = np.array(kernel_size, dtype=np.int32)
    self.num_layers = num_layers
    self.strides = np.array(strides, dtype=np.int32)
    self.output_dims = None

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
      #tf.keras.layers.Dense(self.latent_dim)
    ])

    
    for i in range(0, self.num_layers):
      # eq (12) from Zhao et al., 2017
      self.inference_layers.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(128, activation='relu'),
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
          tf.keras.layers.Dense(128, activation='relu'),
        ])
      )
      self.generative_layers_v.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation='relu'),
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

  def elbo(self, batch, training=False, **kwargs):
    """if 'epoch' in kwargs:
      max_beta = 1
      if 'beta' in kwargs:
        max_beta = kwargs['beta']
      beta = max_beta * min(kwargs['epoch'] / self.warmup_epochs, 1)
    else:
      if 'beta' in kwargs:
        beta = kwargs['beta']
      else:
        beta = 1"""
  
    if 'beta' in kwargs:
      beta = kwargs['beta']
    else:
      beta = 1

    hl_list, mean_list, var_list = self.encode(batch)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=False)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    
    for i in range(self.num_layers):
      if i == 0:
        kl_divergence = compute_kl_divergence_standard_prior(mean_list[0], tf.math.log(var_list[0] + 1e-7))
        kl_divergence = tf.reduce_sum(kl_divergence, axis=1)
      else:
        kl_divergence += tf.reduce_sum(compute_kl_divergence_standard_prior(mean_list[i], tf.math.log(var_list[i] + 1e-7)), axis=1)


    print(tf.reduce_mean(logpx_z).numpy(), tf.reduce_mean(kl_divergence).numpy())
    elbo = tf.reduce_mean(logpx_z - beta * kl_divergence)
    #elbo = tf.reduce_mean(logpx_z - 0 * kl_divergence)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def train_step(self, batch, optimizers, **kwargs):
    with tf.GradientTape() as tape:
      elbo, logpx_z, kl_divergence = self.elbo(batch, training=True, **kwargs)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))
        
      return elbo, logpx_z, kl_divergence

  def forward(self, batch, apply_sigmoid=False):
    hl_list, mean_list, var_list = self.encode(batch)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list)
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

  def encode_across_layers(self, mean_list, var_list):
    z_sample_list = [None] * self.num_layers
    z_hat_list = [None] * self.num_layers
    
    # process the last layer
    index = self.num_layers - 1
    z = self.reparameterize(mean_list[index], var_list[index])
    z_sample_list[index] = z
    # eq (8)
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z)], axis=1))
    
    for index in reversed(range(0, self.num_layers - 1)):
      # z_i (input for layer i) is sampled from the distribution define in layer i + 1.
      # e.g. the prior for the generative distribution mean_p1, var_p1 for layer 1 takes the z sampled from layer 2 as the input.
      z_hat_prior = z_hat_list[index + 1]
      z = self.reparameterize(mean_list[index], var_list[index])
      z_sample_list[index] = z
      # eq (9)
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z)], axis=1))

    return z_sample_list, z_hat_list

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    
    return logits

  def generate(self, z=None, layer=3, num_generated_images=15, **kwargs):
    if z is None:
      z = tf.zeros(shape=(num_generated_images, self.latent_dim))
    
    num_generated_images = z.shape[0]

    z_hat_list = [None] * self.num_layers
    z_sample_list = [tf.zeros(shape=(num_generated_images, self.latent_dim))] * self.num_layers
    z_sample_list[layer - 1] = z

    index = self.num_layers - 1
    z = z_sample_list[index]
    z_hat_list[index] = self.generative_layers_u[index](tf.concat([z, self.generative_layers_v[index](z)], axis=1))
    
    for index in reversed(range(0, self.num_layers - 1)):
      z_hat_prior = z_hat_list[index + 1]
      z = z_sample_list[index]
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, self.generative_layers_v[index](z)], axis=1))

    return self.decode(z_hat_list[0], apply_sigmoid=True)

  def reparameterize(self, mean, var):
    eps = tf.random.normal(shape=mean.shape) # each distribution has its own epsilon
    std = tf.math.sqrt(var)
    return eps * std + mean

  def average_kl_divergence(self, batch, layer=3):
    hl_list, mean_list, var_list = self.encode(batch)
    #z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list)

    return tf.reduce_mean(compute_kl_divergence_standard_prior(mean_list[layer-1], tf.math.log(var_list[layer-1] + 1e-7)), axis=0)