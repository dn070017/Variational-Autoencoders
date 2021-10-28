import tensorflow as tf
from models.vlae import VLAE
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior

class ProVLAE(VLAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_layers=3, num_epoch_per_progress=10, fadein_iter=1500, gamma=0.5, prefix='provlae'):
    super().__init__(latent_dim, input_dims, kernel_size, strides, num_layers, 0, prefix)
    # warmup is not used
    self.fadein_iter = fadein_iter
    self.current_iter = [0] * num_layers
    self.current_iter[num_layers - 1] = fadein_iter
    self.progressive_step = 1
    self.num_epoch_per_progress = num_epoch_per_progress
    self.gamma = gamma

  def train_step(self, batch, optimizers, **kwargs):
    self.progressive_step = min(int((kwargs['epoch'] - 1) / self.num_epoch_per_progress + 1), self.num_layers)
    for l in range(self.num_layers - 1, self.num_layers - self.progressive_step -1, -1):
      self.current_iter[l] += 1

    print(self.progressive_step, self.current_iter)

    with tf.GradientTape() as tape:
      kwargs['training'] = True
      elbo, logpx_z, kl_divergence = self.elbo(batch, **kwargs)
      gradients = tape.gradient(-1 * elbo, self.trainable_variables)
      optimizers['primary'].apply_gradients(zip(gradients, self.trainable_variables))
        
      return elbo, logpx_z, kl_divergence

  def elbo(self, batch, **kwargs):
    training = kwargs['training'] if 'training' in kwargs else False
    beta = kwargs['beta'] if 'beta' in kwargs else 1.0

    hl_list, mean_list, var_list = self.encode(batch, training=training)
    z_sample_list, z_hat_list = self.encode_across_layers(mean_list, var_list, training=training)
    x_pred = self.decode(z_hat_list[0], apply_sigmoid=False)
    
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    
    for i in range(self.num_layers):
      kl_coefficient = beta
      if i < self.num_layers - self.progressive_step:
        kl_coefficient = self.gamma
      if i == 0:
        kl_divergence = compute_kl_divergence_standard_prior(mean_list[0], tf.math.log(var_list[0] + 1e-7))
        kl_divergence = kl_coefficient * tf.reduce_sum(kl_divergence, axis=1)
      else:
        kl_divergence += kl_coefficient * tf.reduce_sum(compute_kl_divergence_standard_prior(mean_list[i], tf.math.log(var_list[i] + 1e-7)), axis=1)

    elbo = tf.reduce_mean(logpx_z - kl_divergence)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

  def encode(self, batch, training=False):
    hl_list = []
    mean_list = []
    var_list = []
    hl_list.append(self.encoder(batch['x']))
    for l in range(1, self.num_layers + 1):
      alpha = min(1.0, self.current_iter[l-1] / self.fadein_iter)

      print('encode:', l, alpha)

      # Eq (12) h_l = gl(h_l-1)
      hl = self.inference_layers[l-1](hl_list[l-1], training=training)
      hl_list.append(hl)
      # Eq (13) mean = mu_l(h_l), var = var_l(h_l)
      mean_list.append(self.inference_mean_linear_layers[l-1](alpha * hl))
      var_list.append(self.inference_var_softplus_layers[l-1](alpha * hl))
    
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
      alpha = min(1.0, self.current_iter[index] / self.fadein_iter)
      
      print('decode:', index + 1, alpha)
      # z_i (input for layer i) is sampled from the distribution define in layer i + 1.
      # e.g. the prior for the generative distribution mean_p1, var_p1 for layer 1 takes the z sampled from layer 2 as the input.
      z_hat_prior = z_hat_list[index + 1]
      z = self.reparameterize(mean_list[index], var_list[index], training=training)
      z_sample_list[index] = z
      # eq (9)
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, alpha * self.generative_layers_v[index](z, training=training)], axis=1), training=training)

    return z_sample_list, z_hat_list

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
      alpha = min(1.0, self.current_iter[index] / self.fadein_iter)
      z_hat_prior = z_hat_list[index + 1]
      z = z_sample_list[index]
      z_hat_list[index] = self.generative_layers_u[index](tf.concat([z_hat_prior, alpha * self.generative_layers_v[index](z, training=False)], axis=1), training=False)

    return self.decode(z_hat_list[0], apply_sigmoid=True)