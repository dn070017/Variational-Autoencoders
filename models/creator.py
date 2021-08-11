#%%
from models.vae import BaseVAE
from models.cvae import CVAE
from models.tcvae import TCVAE
from models.factorvae import FactorVAE
from models.rfvae import RFVAE
import warnings

class VAE(object):
    @staticmethod
    def create_model(model='VAE', kwargs={}):
      model = model.lower()
      if model in ['vae', 'beta-vae', 'bvae', 'b-vae']:
          return BaseVAE(**kwargs)
      elif model in ['tcvae', 'beta-tcvae', 'btcvae', 'b-tcvae']:
          return TCVAE(**kwargs)
      elif model in ['factorvae']:
          return FactorVAE(**kwargs)
      elif model in ['rfvae', 'relevance-factor-vae']:
          return RFVAE(**kwargs)
      elif model in ['cvae', 'conditional-vae']:
          return CVAE(**kwargs)
      else:
          warnings.warn(f'no matched model name can be found for {model}, use beta-vae instead')
          return BaseVAE(**kwargs)

# %%
