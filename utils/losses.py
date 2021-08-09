import numpy as np
import tensorflow as tf

def compute_log_normal_pdf(mu_true, logvar_true, mu_pred):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((mu_pred - mu_true) ** 2. * tf.exp(-logvar_true) + logvar_true + log2pi)

def compute_cross_entropy(p_true, p_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=p_pred, labels=p_true)

def compute_kl_divergence(mean, logvar):
    return -.5 * ((1 + logvar) - tf.exp(logvar) - tf.pow(mean, 2))

def beta_vae_loss(model, x_true, beta=1.0):
    mean, logvar = model.encode(x_true)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)

    cross_ent = compute_cross_entropy(x_true, x_pred)
    logpx_z = -1 * tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    kl_divergence = tf.reduce_sum(compute_kl_divergence(mean, logvar), axis=1)

    total_loss = -tf.reduce_mean(logpx_z - beta * kl_divergence)
    
    return total_loss, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)