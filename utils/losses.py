import numpy as np
import tensorflow as tf

def compute_cross_entropy(p_dist, sample):
    # x: sample
    # p: p_dist
    # -1[xlog(p) + (1 - x)log(1 - p)]
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=p_dist, labels=sample)

def compute_kl_divergence(mean, logvar):
    # Supplmentary B from Kingma et al., 2014 (before applying summation 𝚺_j )
    return -.5 * ((1 + logvar) - tf.exp(logvar) - tf.pow(mean, 2))

def compute_log_normal_pdf(mu, logvar, sample):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((sample - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi)

def compute_log_bernouli_pdf(p_dist, sample):
    # logpx_p = xlog(p) + (1 - x)log(1 - p) 
    return -1 * compute_cross_entropy(p_dist, sample)

def compute_total_correlation(mu_true, logvar_true, mu_pred):
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