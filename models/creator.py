#%%
from models.vae import BaseVAE
from models.tcvae import TCVAE
from models.factorvae import FactorVAE
import warnings

class VAE(object):
    @staticmethod
    def create_model(model='VAE', kwargs={}):
      if model in ['vae', 'beta-vae', 'bvae', 'b-vae']:
          return BaseVAE(**kwargs)
      elif model in ['tcvae', 'beta-tcvae', 'btcvae', 'b-tcvae']:
          return TCVAE(**kwargs)
      elif model in ['factorvae']:
          return FactorVAE(**kwargs)
      else:
          warnings.warn(f'no matched model name can be found for {model}, use beta-vae instead')
          return BaseVAE(**kwargs)

# %%
