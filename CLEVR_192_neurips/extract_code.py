# from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/extract_code.py

import argparse
import pickle

import torch
import numpy as np
import lmdb
from tqdm import tqdm

from vqvae import Model
from utils.utils import *
from data.load_data import *
from utils.config import vqvae_config

batch_size = 32
iterations = 15000


num_hiddens = 128
num_residual_hiddens = 64
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 256
commitment_cost = 0.25

def extract(lmdb_env, loader, model, device):
    index = 0

    pbar = tqdm(loader)

    for imgs, label, bbox, _ in pbar:
        with torch.no_grad():
            with lmdb_env.begin(write=True) as txn:
                imgs = imgs.to(device);
                _, _, sa_quant, norm_quant = model(imgs)
                sa_quant = sa_quant.detach().cpu().numpy()
                norm_quant = norm_quant.detach().cpu().numpy()
                label = label.cpu().numpy()
                bbox = bbox.cpu().numpy()

                for sa_q, norm_q, label, bbox in zip(sa_quant, norm_quant, label, bbox):
                    row = CodeRow(q1=sa_q, q2=norm_q, bboxes=bbox, lbls=label)
                    txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                    index += 1
                    pbar.set_description(f'inserted: {index}')

                txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':

    config = vqvae_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(3, num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, embedding_prior_input=True).to(device)

    with open("models/vqvae/vqvae50000.pt", 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    # Load Data
    training_data, _= get_clevr_data(config)


    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=False,
                             drop_last=False)

    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open("./data/vq_codes_clevr", map_size=map_size)

    extract(env, training_loader, model, device)
