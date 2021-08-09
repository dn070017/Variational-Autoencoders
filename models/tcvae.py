from models.vae import VAE
from utils.losses import tcvae_loss

class TCVAE(VAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), name='tcvae'):
    super(VAE, self).__init__()
