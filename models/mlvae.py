import numpy as np
import tensorflow as tf

from models.betavae import BetaVAE
from utils.losses import compute_log_bernouli_pdf, compute_kl_divergence_standard_prior
from utils.utils import compute_output_dims

class MLVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), latent_content_dim=2, prefix='tcvae'):
    super(MLVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.latent_content_dim = latent_content_dim
    self.content_encoder = tf.keras.Sequential([
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
      tf.keras.layers.Dense(2 * self.latent_content_dim)
    ])
    self.unique_content_mean = None
    self.unique_content_logvar = None

  def elbo(self, batch, **kwargs):
    beta = kwargs['beta'] if 'beta' in kwargs else 1.0
    content_mean, content_logvar, style_mean, style_logvar = self.encode(batch)
    content_mean, content_logvar, unique_content_mean, unique_content_logvar = self.accumulate_group_evidence(content_mean, content_logvar, batch)
    
    mean = tf.concat([content_mean, style_mean], axis=1)
    logvar = tf.concat([content_logvar, style_logvar], axis=1)

    z = self.reparameterize(mean, logvar)
    x_pred = self.decode(z, apply_sigmoid=False)

    num_content = unique_content_mean.shape[0]
    
    # i is the index for x in all X (index for batch['x'])
    # j is the index for x in content Xg (index for batch['x'][Xg])
    # c is the index for content in group G (index for unique_content)
    # 𝚺_G[𝚺_j[.]] = 𝚺i[.]

    # ELBO = 1/G * 𝚺_G[𝚺_j[E_c,Xg[[E_sj_xj[logpxj_c,sj]]]] - 𝚺_j[KL(qsj_xj||ps)] - KL(qc_Xg||c)]
    #      = 1/G * [𝚺_G[𝚺_j[E_c,Xg[[E_sj_xj[logpxj_c,sj]]]] - 𝚺_G[𝚺_j[KL(qsj_xj||ps)]] - 𝚺_G[KL(qc_Xg||c)]]
    #      = 1/G * [𝚺_i[E_c,Xg[[E_sj_xj[logpxj_c,sj]]]] - 𝚺_i[KL(qsj_xj||ps)] - 𝚺_G[KL(qc_Xg||c)]]

    # E_c,Xg[[E_sj_xj[logpxj_c,sj]]]
    logpx_z = compute_log_bernouli_pdf(x_pred, batch['x'])
    logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    # 𝚺_i[E_c,Xg[[E_sj_xj[logpxj_c,sj]]]]
    sum_logpx_z = tf.reduce_sum(logpx_z, axis=0)

    # KL(qs_xi||ps)
    kl_qs_xi_ps = tf.reduce_sum(compute_kl_divergence_standard_prior(style_mean, style_logvar), axis=-1)
    # 𝚺_i[KL(qsj_xj||ps)]
    sum_kl_qs_xi_ps = tf.reduce_sum(kl_qs_xi_ps, axis=0)

    # KL(qc_Xg||c)
    kl_qc_xg_c = tf.reduce_sum(compute_kl_divergence_standard_prior(content_mean, content_logvar), axis=-1)
    # 𝚺_G[KL(qc_Xg||c)]]
    sum_kl_qc_xg_c = tf.reduce_sum(kl_qc_xg_c, axis=0)

    elbo = 1/num_content * (sum_logpx_z - sum_kl_qs_xi_ps - sum_kl_qc_xg_c)

    return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_qs_xi_ps) + tf.reduce_mean(kl_qc_xg_c)


  def forward(self, batch, apply_sigmoid=False):
    content_mean, content_logvar, style_mean, style_logvar = self.encode(batch)
    content_mean, content_logvar, unique_content_mean, unique_content_logvar = self.accumulate_group_evidence(content_mean, content_logvar, batch)
    
    mean = tf.concat([content_mean, style_mean], axis=1)
    logvar = tf.concat([content_logvar, style_logvar], axis=1)

    z = self.reparameterize(mean, logvar)
    x_pred = self.decode(z, apply_sigmoid=apply_sigmoid)
  
    return mean, logvar, z, x_pred

  def encode(self, batch):
    content_mean, content_logvar = tf.split(self.content_encoder(batch['x']), num_or_size_splits=2, axis=1)
    style_mean, style_logvar = tf.split(self.encoder(batch['x']), num_or_size_splits=2, axis=1)
    #mean = tf.concat([group_mean, style_mean], axis=1)
    #logvar = tf.concat([group_logvar, style_logvar], axis=1)
    return content_mean, content_logvar, style_mean, style_logvar

  def accumulate_group_evidence(self, content_mean, content_logvar, batch, save_content=True):
    unique_content_mean = []
    unique_content_logvar = []
    for i in tf.range(batch['y'].shape[1]):
      Xg_cond = batch['y'][:, i] == 1.0
      Xg_indices = tf.where(Xg_cond)
      if Xg_indices.shape[0] == 0:
        continue
      
      # get the sample mean and logvar that correspond to group i 
      content_batch_mean = tf.squeeze(tf.gather(content_mean, Xg_indices, axis=0))
      content_batch_logvar = tf.squeeze(tf.gather(content_logvar, Xg_indices, axis=0))
      
      # reshape to (num_samples_in_group_i, latent_group_dim)
      content_batch_mean = tf.reshape(content_batch_mean, (-1, self.latent_content_dim))
      content_batch_logvar = tf.reshape(content_batch_logvar, (-1, self.latent_content_dim))

      # transform from logvar to var (+1e-7 to avoid negative value with tf.math.log) 
      group_batch_var = tf.exp(content_batch_logvar) + 1e-7

      # accumulate mean and logvar with equation (7) from Bouchacourt et al. 2018.
      # (can potentially do this using tensorflow_probability)
      acc_group_batch_var = 1 / tf.reduce_sum(1 / group_batch_var, axis=0)
      acc_group_batch_mean = tf.expand_dims(tf.reduce_sum(content_batch_mean * 1 / group_batch_var, axis=0) * acc_group_batch_var, 0)
      acc_group_batch_logvar = tf.expand_dims(tf.math.log(acc_group_batch_var), 0)

      # reassign the group mean and logvar back to the samples
      content_mean = tf.where(tf.expand_dims(Xg_cond, axis=-1), acc_group_batch_mean, content_mean)
      content_logvar = tf.where(tf.expand_dims(Xg_cond, axis=-1), acc_group_batch_logvar, content_logvar)

      # save the accumulated group mean and logvar for each group
      # for more efficient computation for ELBO.
      # acc_mean_unique[i]: the merge mean for group i
      # acc_logvar_unique[i]: the merge logvar for group i
      # shape: (num_group, latent_group_dim)
      unique_content_mean.append(acc_group_batch_mean)
      unique_content_logvar.append(acc_group_batch_logvar)

    unique_content_mean = tf.concat(unique_content_mean, axis=0)
    unique_content_logvar = tf.concat(unique_content_logvar, axis=0)

    if save_content:
      self.unique_content_mean = unique_content_mean
      self.unique_content_logvar = unique_content_logvar

    return content_mean, content_logvar, unique_content_mean, unique_content_logvar

  def average_kl_divergence(self, batch):
    group_mean, group_logvar, style_mean, style_logvar = self.encode(batch)
    group_mean, group_logvar, unique_content_mean, unique_content_logvar = self.accumulate_group_evidence(group_mean, group_logvar, batch)

    content_kl_divergence = compute_kl_divergence_standard_prior(unique_content_mean, unique_content_logvar)
    content_kl_divergence = tf.reduce_mean(content_kl_divergence, axis=0)

    style_kl_divergence = tf.reduce_mean(compute_kl_divergence_standard_prior(style_mean, style_logvar), axis=0)

    return tf.squeeze(tf.concat([content_kl_divergence, style_kl_divergence], axis=-1))

  def generate(self, z=None, num_generated_images=15, **kwargs):
    if z is None:
      z = tf.random.normal(shape=(num_generated_images, self.latent_dim), dtype=tf.float32)

    num_samples = z.shape[0]

    if 'content' not in kwargs and 'target' not in kwargs:
      target = 0
      z_content = []
      for i in range(num_samples):
        target += 1
        if target >= self.unique_content_mean.shape[0]:
          target = 0
        z_content.append(self.unique_content_mean[target])
      z_content = tf.concat(z_content, axis=0)
    elif 'target' in kwargs:
      target = kwargs['target']
      z_content = tf.repeat(
        tf.expand_dims(self.unique_content_mean[target], 0),
        repeats=num_samples,
        axis=0
      )
    elif 'content' in kwargs:
      z_content = kwargs['content']
    
    return self.decode(tf.concat([z_content, z[:, self.latent_content_dim:]], axis=1), apply_sigmoid=True)