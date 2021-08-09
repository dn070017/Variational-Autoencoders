  def train_step(model, x_true, optimizer, beta=1.0):
    with tf.GradientTape() as tape:
      total_loss, reconstructed_loss, kl_divergence = beta_vae_loss(model, x_true, beta)
      gradients = tape.gradient(total_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, reconstructed_loss, kl_divergence