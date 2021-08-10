import numpy as np
import tensorflow as tf

def compute_log_normal_pdf(mu_true, logvar_true, mu_pred):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((mu_pred - mu_true) ** 2. * tf.exp(-logvar_true) + logvar_true + log2pi)

def compute_cross_entropy(p_true, p_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=p_pred, labels=p_true)

def tcvae_batch_estimator(q):
    # E_q[log(q(z))] ~ 1/M 𝚺[log(𝚺q(z_i|z_j)) - log(NM)] ignored log(NM)
    # E_q[log(q(z))] ~ 1/M 𝚺[log(𝚺q(z_i|z_j))]
    return 

def compute_kl_divergence(mean, logvar):
    return -.5 * ((1 + logvar) - tf.exp(logvar) - tf.pow(mean, 2))

def compute_tc_loss(mu_true, logvar_true, mu_pred):
    # i: index of sampling point from the distribution of sample i.
    # j: index of distribution inferred from sample j.
    # k: index of the latent dimensions
    # E_q[log(q(z) / 𝚷_k(q(z_k))] (ii) in eq(2)
    # = E_q[log(q(z)) - 𝚺_k[log(q(z_k))]]

    # log[q(z_i|z_j)]: logqz_i_j
    # shape: (M, M, D)
    logqz_i_j = compute_log_normal_pdf(tf.expand_dims(mu_true, 0), tf.expand_dims(logvar_true, 0), tf.expand_dims(mu_pred, 1))

    # joint distribution: q(z_ij1, z_ij2...)
    # log(q(z)): logqz
    # log(q(z)) ~ log(𝚺_j𝚺_k q(z_ik|z_jk)) - log(NM) ignored log(NM)
    # log(q(z)) ~ log(𝚺_j𝚺_k q(z_ik|z_jk))
    logqz = tf.reduce_logsumexp(tf.reduce_sum(logqz_i_j, axis=2), axis=1)

    # independent distribution: q(z_ij1)q(z_ij2)... 
    # 𝚺_k[log(q(z_k))]: sigma_logq_k
    # 𝚺_k[log(q(z))] ~ 𝚺_k[log(𝚺_j q(z_ik|z_jk))]
    sigma_logq_k = tf.reduce_sum(tf.reduce_logsumexp(logqz_i_j, axis=1), axis=1)

    return logqz - sigma_logq_k

def vae_loss(model, x_true, beta=1.0):
    mean, logvar = model.encode(x_true)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)

    cross_ent = compute_cross_entropy(x_true, x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)

    total_loss = -tf.reduce_mean(logpx_z - beta * kl_divergence)
    
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def tcvae_loss(model, x_true, beta=1.0):
    mean, logvar = model.encode(x_true)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)

    cross_ent = compute_cross_entropy(x_true, x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    tc_loss = compute_tc_loss(mean, logvar, z)

    total_loss = -tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))
    
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)

def factorvae_loss(model, x_true, beta=1.0):
    mean, logvar = model.encode(x_true)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)

    density = model.discriminator(z)

    cross_ent = compute_cross_entropy(x_true, x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)
    tc_loss = tf.reduce_mean(density[:, 0] - density[:, 1])

    total_loss = -tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))

    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)