import numpy as np
import tensorflow as tf
from utils.utils import compute_output_dims
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior, compute_kl_divergence

class LVAE(tf.keras.Model):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_ladders=2, warmup_epochs=25, prefix='vae'):
    super().__init__()
    self.prefix = prefix
    #self.latent_dim = (latent_dim // num_ladders) * num_ladders
    #self.latent_dim = np.array(latent_dim // num_ladders, dtype=np.int32)
    self.latent_dim = latent_dim
    self.input_dims = np.array(input_dims, dtype=np.int32)
    self.kernel_size = np.array(kernel_size, dtype=np.int32)
    self.num_ladders = num_ladders
    self.strides = np.array(strides, dtype=np.int32)
    self.output_dims = None
    
    self.warmup_epochs = warmup_epochs

    self.mlp_layers = []
    self.mean_linear_layers = []
    self.var_softplus_layers = []
    

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

    
    for i in range(0, self.num_ladders):
      # eq (13) from Sønderby et al 2016
      # p(z_l|x) is parameterized using fully connected neural network (l = 1...L)
      self.mlp_layers.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(self.latent_dim),
        ])
      )
      # eq (14) from Sønderby et al 2016
      self.mean_linear_layers.append(
        tf.keras.layers.Dense(self.latent_dim)
      )
      # eq (15) from Sønderby et al 2016
      self.var_softplus_layers.append(
        tf.keras.Sequential([
          tf.keras.layers.Dense(self.latent_dim),
          tf.keras.layers.Lambda(lambda x: tf.math.softplus(x))
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
      max_beta = 1
      if 'beta' in kwargs:
        max_beta = kwargs['beta']
      beta = max_beta * min(kwargs['epoch'] / self.warmup_epochs, 1)

    dn_list, mean_hat_list, var_hat_list = self.encode(batch, training=training)
    mean_p_list, var_p_list, mean_q_list, var_q_list, z_sample_list = self.encode_across_ladder(mean_hat_list, var_hat_list)
    x_pred = self.decode(z_sample_list[0], apply_sigmoid=False)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    
    kl_divergence = 0
    for mean_p, var_p, mean_q, var_q in zip(mean_p_list, var_p_list, mean_q_list, var_q_list):
      logvar_q = tf.math.log(var_q + 1e-7)
      logvar_p = tf.math.log(var_p + 1e-7)
      kl_divergence += tf.reduce_sum(compute_kl_divergence(mean_q, mean_p, logvar_q, logvar_p), axis=1)

    elbo = tf.reduce_mean(logpx_z - beta * kl_divergence)
    #elbo = tf.reduce_mean(logpx_z - 0 * kl_divergence)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def train_step(self, batch, optimizers, **kwargs):
    with tf.GradientTape() as tape:
      kwargs['training'] = True
      elbo, logpx_z, kl_divergence = self.elbo(batch, **kwargs)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))
        
      return elbo, logpx_z, kl_divergence

  def forward(self, batch, apply_sigmoid=False):
    dn_list, mean_hat_list, var_hat_list = self.encode(batch)
    mean_p_list, var_p_list, mean_q_list, var_q_list, z_sample_list = self.encode_across_ladder(mean_hat_list, var_hat_list)
    x_pred = self.decode(z_sample_list[0], apply_sigmoid=apply_sigmoid)
  
    return tf.concat(mean_q_list, axis=1), tf.concat(var_q_list, axis=1), dn_list, x_pred

  def encode(self, batch, training=False):
    dn_list = []
    mean_hat_list = []
    var_hat_list = []
    dn_list.append(self.encoder(batch['x']))
    for i in range(1, self.num_ladders + 1):
      # Eq (13) di = MLP(d_i-1)
      dn = self.mlp_layers[i-1](dn_list[i-1], training=training)
      dn_list.append(dn)
      # Eq (14) mean_hat_qi = linear(di)
      mean_hat_list.append(self.mean_linear_layers[i-1](dn))
      # Eq (15) var_hat_qi = softplus(linear(di))
      var_hat_list.append(self.var_softplus_layers[i-1](dn))
    
    # If num_ladders = 2
    # dn_list: [x0, d1, d2]
    # mean_hat_list: [mu1, mu2]
    # var_hat_list: [var1, var2]

    return dn_list, mean_hat_list, var_hat_list

  def encode_across_ladder(self, mean_hat_list, var_hat_list):
    z_sample_list = [None] * self.num_ladders
    mean_q_list = [None] * self.num_ladders
    var_q_list = [None] * self.num_ladders
    mean_p_list = [None] * self.num_ladders
    var_p_list = [None] * self.num_ladders
    
    # process the last layer
    index = self.num_ladders - 1
    z = self.reparameterize(mean_hat_list[index], var_hat_list[index])   
    z_sample_list[index] = z
    mean_p = tf.zeros_like(z)
    var_p = tf.ones_like(z)
    mean_q_list[index] = mean_hat_list[index]
    var_q_list[index] = var_hat_list[index]
    mean_p_list[index] = mean_p
    var_p_list[index] = var_p

    for index in reversed(range(0, self.num_ladders - 1)):
      # z_i (input for layer i) is sampled from the distribution define in layer i + 1.
      # e.g. the prior for the generative distribution mean_p1, var_p1 for layer 1 takes the z sampled from layer 2 as the input.
      z_prior = z_sample_list[index + 1]
      mean_p = self.mean_linear_layers[index](z_prior)
      var_p = self.var_softplus_layers[index](z_prior)

      mean_p_list[index] = mean_p
      var_p_list[index] = var_p

      mean_hat = mean_hat_list[index]
      var_hat = var_hat_list[index]

      mean_q, var_q = self.precision_weighted_combination(mean_p, mean_hat, var_p, var_hat)
      
      mean_q_list[index] = mean_q
      var_q_list[index] = var_q

      z = self.reparameterize(mean_q, var_q)
      z_sample_list[index] = z

    return mean_p_list, var_p_list, mean_q_list, var_q_list, z_sample_list

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    
    return logits

  def precision_weighted_combination(self, mean_zn, mean_dn, var_zn, var_dn):
    # eq (18) and eq (19)
    precision_zn = 1 / (var_zn + 1e-7)
    precision_dn = 1 / (var_dn + 1e-7)
    var = tf.square(1 / (precision_zn + precision_dn))
    mean = (mean_zn * precision_zn + mean_dn * precision_dn) / (precision_zn + precision_dn)
    return mean, var

  def generate(self, z=None, ladder=2, num_generated_images=15, **kwargs):
    if z is None:
      z = tf.random.normal(shape=(num_generated_images, self.latent_dim))

    for index in reversed(range(0, ladder)):
      mean = self.mean_linear_layers[index](z)
      var = self.var_softplus_layers[index](z)
      z = self.reparameterize(mean, var)

    return self.decode(z, apply_sigmoid=True)

  def reparameterize(self, mean, var):
    eps = tf.random.normal(shape=mean.shape) # each distribution has its own epsilon
    std = tf.math.sqrt(var)
    return eps * std + mean

  def average_kl_divergence(self, batch, ladder=2):
    dn_list, mean_hat_list, var_hat_list = self.encode(batch)
    mean_p_list, var_p_list, mean_q_list, var_q_list, z_sample_list = self.encode_across_ladder(mean_hat_list, var_hat_list)

    index = ladder - 1

    mean_p = mean_p_list[index]
    mean_q = mean_q_list[index]
    var_p = var_p_list[index]
    var_q = var_q_list[index]

    logvar_q = tf.math.log(var_q + 1e-7)
    logvar_p = tf.math.log(var_p + 1e-7)

    return tf.reduce_mean(compute_kl_divergence(mean_q, mean_p, logvar_q, logvar_p), axis=0)
