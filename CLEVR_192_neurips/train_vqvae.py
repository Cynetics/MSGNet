from __future__ import print_function

import numpy as np

from six.moves import xrange

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.utils import make_grid, save_image
from utils.utils import *
from data.load_data import get_clevr_data
from vqvae import Model
from utils.config import vqvae_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_loader, validation_loader, config):

    print("Started Training!")
    model.train(); model = model.to(device);
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.98)

    train_res_recon_error = []
    train_res_perplexity = []
    train_loss = []
    train_perplexity = []
    for i in xrange(config.iterations):
        imgs, _, _, _ = next(iter(training_loader))
        imgs = imgs.to(device)
        vq_loss, imgs_recon,  norm_perplexity, sa_perplexity = model(imgs)
        recon_error = torch.mean((imgs_recon - imgs)**2) / torch.var(imgs)
        optimizer.zero_grad()
        loss = recon_error + vq_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_res_recon_error.append(recon_error.item())
        train_loss.append(vq_loss.item())
        train_perplexity.append(sa_perplexity.item())

        if (i+1) % 250 == 0:
            print("Current Iteration: {}".format(i+1))
            print("Perplexity of the normal path: ", norm_perplexity.item())
            print("Perplexity of the attention: ", sa_perplexity.item())
            print('Reconstruction Error: %.4f' % np.mean(train_res_recon_error[-100:]))
            print('VQVAE Loss: %.8f' % np.mean(train_loss[-100:]))
            print("Learning Rate: ", optimizer.param_groups[0]['lr'])
            print()

        if (i+1) % 1000 == 0:
            valid_originals, valid_reconstructions = reconstruct(model, validation_loader, device)

            save_image(make_grid((valid_originals.detach().cpu()*0.5)+0.5), "./reconstructions/originals/Originals_{}.png".format(i), normalize=True)
            save_image(make_grid((valid_reconstructions.detach().cpu()*0.5)+0.5), "./reconstructions/reconstructions/recon_{}.png".format(i), normalize=True)
            print("Saving Model...")
            torch.save(model.state_dict(),"models/vqvae/vqvae{}.pt".format(i+1))
            model.train()

    return train_res_recon_error, train_res_perplexity


def main():

    config = vqvae_config()

    model = Model(3, config.num_hiddens, config.n_res_block, config.n_res_hiddens,
              config.n_embedding, config.embedding_dim,
              config.commitment_cost,embedding_prior_input=False)

    # Load Data
    training_data, validation_data = get_clevr_data(config)


    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=True)
    validation_loader = DataLoader(validation_data,
                               batch_size=16,
                               shuffle=True)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

    model = model.to(device)
    # Train
    recon_error, perplexity = train(model, training_loader, validation_loader, config)

    # Reconstruct the trained model
    valid_originals, valid_reconstructions = reconstruct(model, validation_loader, device)

    # save reconstructions
    save_image(make_grid((valid_originals.cpu()*0.5)+0.5), "Finished_Originals.png", normalize=True)
    save_image(make_grid((valid_reconstructions.cpu()*0.5)+0.5), "Finished_Reconstructions.png", normalize=True)


if __name__ == '__main__':
    main()
