#%%
import argparse
import os
import numpy as np
import tensorflow as tf
import warnings

from models.creator import VAE
from utils.utils import preprocess_images, generate_and_save_images, animated_image_generation, plot_latent_images
from utils.losses import vae_loss
from time import time
from IPython import display

def parse_arguments():
    parser = argparse.ArgumentParser(description='Variational Autoencoders on MNIST Dataset.')
    parser.add_argument(
        '--model', type=str, default='vae',
        help="one of the following: 'vae', 'beta-vae', 'tcvae', 'factorvae', 'rfvae'")
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='coefficient for KL divergence used in beta-VAE (Higgins et al., 2017) or coefficient used for total correlation in beta-TCVAE (Chen, et al., 2018) and in factor-vae (γ) (Kim and Mnih 2019), default: (1.0)')
    parser.add_argument(
        '--num_epochs', type=int, default=25, dest='num_epochs',
        help='maximum number of epochs used during training (default: 25)')
    parser.add_argument(
        '--batch_size', type=int, default=32, dest='batch_size',
        help='batch_size used during training (default: 32)')
    parser.add_argument(
        '--train_size', type=int, default=6400, dest='train_size',
        help='number of samples used during training (default: 6400)')
    parser.add_argument(
        '--num_generated_image', type=int, default=25, dest='test_size',
        help='number of generated images (default: 16)')
    parser.add_argument(
        '--latent_dim', type=int, default=2, dest='latent_dim',
        help='number of latent embeddings (default: 2)')
    parser.add_argument(
        '--prefix', type=str, default='vae', dest='prefix',
        help='directory (default: vae)')
    parser.add_argument(
        '--outdir', type=str, default='./tmp/', dest='outdir',
        help='directory (default: ./tmp/)')

    args = parser.parse_args()
    return args

def main(model_name, beta, num_epochs, train_size, batch_size, latent_dim, test_size, outdir, prefix, show_images):
    test_size = int(np.sqrt(test_size)) ** 2

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    os.makedirs(outdir, exist_ok=True)

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                    .shuffle(train_size).take(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).take(test_size).batch(test_size))

    optimizer_vae = tf.keras.optimizers.Adam(1e-4)
    optimizer_d = tf.keras.optimizers.Adam(1e-4)

    model = VAE.create_model(model_name, {'latent_dim': latent_dim, 'prefix': prefix})

    for epoch in range(1, num_epochs + 1):
        start_time = time()
        for train_x in train_dataset:
            if type(model).__name__ not in ['FactorVAE', 'RFVAE']:
                total_loss, reconstructed_loss, kl_divergence = model.train_step(
                    train_x,
                    optimizer_vae,
                    beta=beta
                )
            else:
                total_loss, reconstructed_loss, kl_divergence = model.train_step(
                    train_x,
                    optimizer_vae,
                    optimizer_d,
                    beta=beta
                )
        end_time = time()
        display.clear_output(wait=False)
        for test_x in test_dataset:
            total_loss, reconstructed_loss, kl_divergence = vae_loss(model, test_x)

        message = f'Epoch: {epoch:>5}   Test ELBO: {-total_loss:>.2f}    Test Reconstructed Loss: {-reconstructed_loss:>.5f}    Test KL-Divergence: {kl_divergence:>.2f}    Time Elapse: {end_time - start_time:.3f}'

        print(message)
        generate_and_save_images(model, outdir, epoch, test_x, show_images)

    animated_image_generation(model, outdir)
    plot_latent_images(model, train_x, outdir, 15, show_images=show_images)

    return model

#%%
if __name__ == "__main__":
    try:
        args = parse_arguments()
        # from command line/ debugger (in VSCODE)
        model = main(
            args.model,
            args.beta,
            args.num_epochs,
            args.train_size,
            args.batch_size,
            args.latent_dim,
            args.test_size,
            args.outdir,
            args.prefix,
            show_images=False
        )
    except:
        # from iPython (in VSCODE)
        warnings.warn('there is an error in the argument, use default parameters instead')
        model = main(
            model_name='rfvae',
            beta=2.0,
            num_epochs=25,
            train_size=6400,
            batch_size=32,
            latent_dim=2,
            test_size=25,
            outdir='tmp',
            prefix='rfvae',
            show_images=True
        )


# %%
