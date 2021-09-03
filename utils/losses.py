import numpy as np
import tensorflow as tf

def compute_log_normal_pdf(mu_true, logvar_true, mu_pred):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((mu_pred - mu_true) ** 2. * tf.exp(-logvar_true) + logvar_true + log2pi)

def compute_cross_entropy(p_true, p_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=p_pred, labels=p_true)

def compute_kl_divergence(mean, logvar):
    return -.5 * ((1 + logvar) - tf.exp(logvar) - tf.pow(mean, 2))

def compute_tc_loss(mu_true, logvar_true, mu_pred):
    # i: index of sampling point from the distribution of sample i.
    # j: index of distribution inferred from sample j.
    # k: index of the latent dimensions
    # E_q[log(q(z) / 𝚷_k(q(z_k))] # (ii) in eq(2) from Chen et al., 2018
    # = E_q[log(q(z)) - 𝚺_k[log(q(z_k))]]

    # log[q(z_i|z_j)]: logqz_i_j
    # shape: (M, M, D)
    logqz_i_j = compute_log_normal_pdf(tf.expand_dims(mu_true, 0), tf.expand_dims(logvar_true, 0), tf.expand_dims(mu_pred, 1))

    # joint distribution: q(z_ij1, z_ij2...)
    # log(q(z)): logqz
    # log(q(z)) ~ log(𝚺_j𝚺_k q(z_ik|z_jk)) - log(NM) #  eq S4 from Chen et al., 2018
    # log(q(z)) ~ log(𝚺_j𝚺_k q(z_ik|z_jk)) # ignored log(NM)
    logqz = tf.reduce_logsumexp(tf.reduce_sum(logqz_i_j, axis=2), axis=1)

    # independent distribution: q(z_ij1)q(z_ij2)... 
    # 𝚺_k[log(q(z_k))]: sigma_logq_k
    # 𝚺_k[log(q(z))] ~ 𝚺_k[log(𝚺_j q(z_ik|z_jk))]
    sigma_logq_k = tf.reduce_sum(tf.reduce_logsumexp(logqz_i_j, axis=1), axis=1)

    return logqz - sigma_logq_k

def vae_loss(model, batch, beta=1.0):
    mean, logvar, z, x_pred = model.forward(batch)

    cross_ent = compute_cross_entropy(batch['x'], x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    total_loss = -tf.reduce_mean(logpx_z - beta * kl_divergence)
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def tcvae_loss(model, batch, beta=1.0):
    mean, logvar, z, x_pred = model.forward(batch)

    cross_ent = compute_cross_entropy(batch['x'], x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    tc_loss = compute_tc_loss(mean, logvar, z)

    total_loss = -tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))
    
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def cvae_loss(model, batch, beta=1.0):
    mean, logvar, z, x_pred = model.forward(batch)

    cross_ent = compute_cross_entropy(batch['x'], x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    total_loss = -tf.reduce_mean(logpx_z - beta * kl_divergence)
    
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def mlvae_loss(model, batch, beta=1.0):
    mean, logvar = model.encode(batch)
    mean, logvar, unique_mean, unique_logvar = model.accumulate_group_evidence(mean, logvar, batch)

    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z, apply_sigmoid=False)

    style_start_idx = model.latent_group_dim
    style_mean = mean[:, style_start_idx:]
    style_logvar = logvar[:, style_start_idx:]
    style_kl_divergence = tf.reduce_sum(compute_kl_divergence(style_mean, style_logvar), axis=1)

    logpx_z = -1 * tf.reduce_sum((batch['x'] - x_pred) ** 2, axis=1)

    group_kl_divergence = tf.reduce_sum(compute_kl_divergence(unique_mean, unique_logvar), axis=1)
    group_kl_divergence = tf.reduce_sum(group_kl_divergence)

    total_loss = (-tf.reduce_sum(logpx_z - (style_kl_divergence)) + group_kl_divergence) / unique_mean.shape[0]

    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(group_kl_divergence)

def factorvae_loss(model, batch, beta=1.0):
    mean, logvar, z, x_pred = model.forward(batch)

    density = model.discriminator(z)

    cross_ent = compute_cross_entropy(batch['x'], x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    tc_loss = tf.reduce_mean(density[:, 0] - density[:, 1])

    total_loss = -tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))

    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def rfvae_loss(model, batch, beta=1.0):
    mean, logvar, z, x_pred = model.forward(batch)

    rc = model.relevance.relevance_coefficient()       # R
    rc_penalty = model.relevance.penalty_coefficient() # lambda

    density = model.discriminator(rc * z)

    cross_ent = compute_cross_entropy(batch['x'], x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    kl_divergence = tf.reduce_sum(rc_penalty * compute_kl_divergence(mean, logvar), axis=1)
    tc_loss = tf.reduce_mean(density[:, 0] - density[:, 1])

    fractional_loss = -1 * tf.reduce_sum(rc * tf.math.log(rc + 1e-7) + (1 - rc) * tf.math.log((1 - rc) + 1e-7))

    total_loss = -tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss)) + model.eta_s * tf.reduce_sum(tf.abs(rc)) + model.eta_h * fractional_loss

    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

    
def soft_introvae_encoder_loss(model, batch, alpha=2.0, beta=1.0):

    s = tf.cast(tf.reduce_prod(batch['x'].shape), dtype=tf.float32)
    mean_x, logvar_x, z_x, x_r = model.forward(batch) # line 4, 5, 8 in Algorithm 1

    z_f = tf.random.normal(shape=z_x.shape) # line 6 in Algorithm 1
    x_f = model.decode(z_f) # line 8 in Algorithm 1
    mean_ff, logvar_ff, z_ff, x_ff = model.forward({'x': x_f}) # line 9, line 10 in Algorithm 1

    cross_ent_x = compute_cross_entropy(batch['x'], x_r)
    logpx_z = -1 * tf.reduce_sum(cross_ent_x, axis=[1, 2, 3])
    cross_ent_f = compute_cross_entropy(batch['x'], x_ff)
    logpf_z = -1 * tf.reduce_sum(cross_ent_f, axis=[1, 2, 3])

    kl_divergence_x = tf.reduce_sum(compute_kl_divergence(mean_x, logvar_x), axis=1)
    kl_divergence_f = tf.reduce_sum(compute_kl_divergence(mean_ff, logvar_ff), axis=1)

    elbo_x = -s * tf.reduce_mean(logpx_z - beta * kl_divergence_x) # line 11 in Algorithm 1
    elbo_f = -tf.reduce_mean(logpf_z - beta * kl_divergence_f) # line 12 in Algorithm 1

    loss_encoder = elbo_x + 1 / alpha * tf.math.exp(alpha / s * elbo_f) # line 13, line 14 in Algorithm 1

    return loss_encoder, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence_x)

def soft_introvae_decoder_loss(model, batch, beta=1.0):

    s = tf.cast(tf.reduce_prod(batch['x'].shape), dtype=tf.float32)
    mean_x, logvar_x, z_x, x_r = model.forward(batch) # line 18 in Algotihm 1
    z_f = tf.random.normal(shape=z_x.shape) 
    x_f = model.decode(z_f) # line 19 in Algorithm 1
    mean_ff, logvar_ff, z_ff, x_ff = model.forward({'x': x_f}) # line 20 in Algorithm 1
    x_ff = tf.stop_gradient(x_ff)

    cross_ent_x = compute_cross_entropy(batch['x'], x_r)
    logpx_z = -1 * tf.reduce_sum(cross_ent_x, axis=[1, 2, 3]) 
  
    cross_ent_f = compute_cross_entropy(batch['x'], x_ff)
    logpf_z = -1 * tf.reduce_sum(cross_ent_f, axis=[1, 2, 3])

    kl_divergence_f = tf.reduce_sum(compute_kl_divergence(mean_ff, logvar_ff), axis=1)
    elbo = -tf.reduce_mean(logpx_z) # line 21 in Algorithm 1
    elbo_f = -tf.reduce_mean(logpf_z - beta * kl_divergence_f) # line 22 in Algorithm 1
    
    loss_encoder = s * (elbo + elbo_f)

    return loss_encoder