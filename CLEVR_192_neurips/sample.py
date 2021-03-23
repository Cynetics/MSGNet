import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.utils import make_grid
from vqvae import Model
from pixelsnail import PixelSNAIL
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils.utils import *
from data.load_data import *
from utils.config import vqvae_config


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row

def load_model(model, checkpoint, device):
    ckpt = torch.load(checkpoint)

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        embedding_dim = 64
        num_embeddings = 256
        
        model = Model(3,num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              0.25, embedding_prior_input=True)

    if model == 'label_model':
        model = PixelSNAIL(
            [24, 24],
            25,
            args.channel,
            5,
            4,
            args.n_res_block,
           args.n_res_channel,
            attention=(True if args.self_attention==1 else False),
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.cond_res_channel,
            n_class=25,
            cond_hidden_dim=args.cond_hidden_dim
        )

    if model == 'pixelsnail':
        model = PixelSNAIL(
            [48, 24],
            args.n_embedding,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=(True if args.self_attention==1 else False),
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.cond_res_channel,
            n_class=25,
            cond_hidden_dim=args.cond_hidden_dim
        )
        #vqvae = vqvae

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

def sample_batch(vqvae,model_top, layout_model, validation_loader, max_objects=2):
    (img,labels, boxes, _) = next(iter(validation_loader))


    img = img.to(device)
    masks = sample_model(layout_model, device, args.batch_size, [24,24], 0.75).float()
    masks = torch.cat([masks, masks], dim=1).long().to(device)
    
    # Uncomment to generate without the Layout PixelSNAIL
    #labels = torch.argmax(labels.to(device),dim=2)
    #masks = get_masks(boxes, labels, 24)
    #masks = torch.cat([masks, masks], dim=1).long().to(device)

    B, _,_, _ = img.shape

    latent = sample_model(model_top, device, args.batch_size, [48, 24], args.temp, condition=masks)
    sa_q, norm_q = latent[:,:24, :], latent[:,24:,:]
    reconstruction = vqvae.decode_latents(sa_q, norm_q, discrete=True)

    save_image(reconstruction, "generation.png", normalize=True, range=(-1, 1))
    show(make_grid((reconstruction.cpu().data*0.5))+0.5)
    show(make_grid(img.cpu().data*0.5)+0.5)

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vqvae', type=str, default="models/vqvae/vqvae_model.pt")
    parser.add_argument('--pixelsnail', type=str, default="models/pixelsnail/clevr_pixelsnail.pt")
    parser.add_argument('--label_model', type=str, default="models/pixelsnail_layout/clevr_layoutpixelsnail.pt")
    parser.add_argument('--temp', type=float, default=0.85)
    parser.add_argument('--filename', type=str, default="generation.png")

    args = parser.parse_args()
    config = vqvae_config()

    vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail', args.pixelsnail, device)
    layout_model = load_model('label_model', args.label_model, device)

    # Load Data
    training_data, _ = get_clevr_data(config)

    validation_loader = DataLoader(validation_data,
                               batch_size=args.batch_size,
                               shuffle=True)

    sample_batch(vqvae, model_top,layout_model,validation_loader)
