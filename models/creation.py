#%%
from models.vae import BaseVAE
from models.tcvae import TCVAE
import warnings

class VAE(object):
    @staticmethod
    def create_model(model='VAE', kwargs={}):
      if model in ['vae', 'beta-vae', 'bvae', 'b-vae']:
          return BaseVAE(**kwargs)
      elif model in ['tcvae', 'beta-tcvae', 'btcvae', 'b-tcvae']:
          return TCVAE(**kwargs)
      else:
          warnings.warn('no matched model name can be found for {model}, use beta-vae instead')
          return BaseVAE(**kwargs)

# %%
