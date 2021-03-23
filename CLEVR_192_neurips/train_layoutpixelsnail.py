import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ClevrData import LMDBDataset

from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler
from utils.utils import *
#from numpy import load

def train(args, epoch, loader, dataset, model, optimizer, scheduler, device):
    loader = tqdm(loader)
    weight = torch.ones(25).to(device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss_list = []
    accuracy_list = []
    for i, (_, _, labels, boxes) in enumerate(loader):

        boxes = boxes.to(device); labels = labels.to(device)

        labels = labels.to(device)
        labels = torch.argmax(labels, dim=2)
        masks = get_masks(boxes, labels, 24).to(device)

        model.zero_grad()
        target = masks
        out, _ = model(masks)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        lr = optimizer.param_groups[0]['lr']
        accuracy_list.append(accuracy.item())
        loss_list.append(loss.item())

        if (i+1) % 1 == 0:
            loader.set_description(
                (
                    f'epoch: {epoch + 1}; loss: {np.mean(loss_list[-100:]):.5f}; '
                    f'acc: {np.mean(accuracy_list[-100:]):.5f}; lr: {lr:.5f}'
                )
            )

    return accuracy_list[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_embedding', type=float, default=25)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--n_res_block', type=int, default=2)
    parser.add_argument('--n_res_channel', type=int, default=128)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=0)
    parser.add_argument('--cond_res_channel', type=int, default=64)
    parser.add_argument('--cond_hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--print_every', type=int, default=300)
    parser.add_argument('--self_attention', type=int, default=1)

    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str, default=None)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--path', type=str, default="./data/vq_codes_clevr")

    args = parser.parse_args()

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LMDBDataset(args.path)

    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, drop_last=True
    )

    ckpt = {}
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    model = PixelSNAIL(
            [24, 24],
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

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(loader) * args.epochs, momentum=None)
    for i in range(args.epochs):
        current_accuracy = train(args, i, loader, dataset, model, optimizer, scheduler, device)
        current_accuracy = round(current_accuracy, 2)
        print("Last Epoch's batch Accuracy: {}".format(current_accuracy))
        if (i+1)%1==0:
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f'models/pixelsnail/mnist_pairs_pixelsnail_{str(i + 1).zfill(3)}_{str(current_accuracy)}.pt',
            )
