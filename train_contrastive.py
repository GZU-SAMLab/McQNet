import os
import sys
import time
from datetime import datetime
import argparse
from copy import deepcopy
import glob
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json

# imports from my own script
import utils
from models.contrast import ContrastResNet

utils.make_deterministic(123)

def setup_args():
    parser = argparse.ArgumentParser('argument for training')

    # general
    parser.add_argument('--eval-freq', type=int, default=1, help="evaluate every this epochs")
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
    parser.add_argument('--workers', type=int, default=4, help="number of processes to make batch worker.")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs. if 0, evaluation only mode")
    parser.add_argument('--gpu', type=int, default=0)

    # optimization
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate. default  is 0.001")
    parser.add_argument('--steps', default=[5], nargs='+', type=int, help='decrease lr at this point')
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to decrease learning rate")

    # dataset
    parser.add_argument("--dataset", choices=["cub", "car", "dog", "nab","miniimagenet"],
                        default="cub",
                        help="Which dataset.")
    parser.add_argument('--backbone', type=str, default="ViT", choices=["ViT", "swin", "resnet18"],
                        help="feature extraction newtork")
    parser.add_argument('--backbone-pretrained', type=int, default=1,
                        help="use pretrained model or not for feature extraction network")

    # folder/route
    parser.add_argument('--dataset_root', type=str, default=None,
                        help="Default is None, and ../data/<datasetname> is used")

    ### meta setting
    parser.add_argument('--nway', default=5, type=int,
                        help='class num to classify for training. this has to be more than 1 and maximum is the total number of classes')
    parser.add_argument('--nway-eval', default=5, type=int,
                        help='class num to classify for evaluation. this has to be more than 1 and maximum is the total number of classes')
    parser.add_argument('--nshot', default=5, type=int, help='number of labeled data in each class, same as nsupport')
    parser.add_argument('--nquery', default=15, type=int, help='number of query point per class')

    parser.add_argument('--episodes-train', type=int, default=1000, help="number of episodes per epoch for train")
    parser.add_argument('--episodes-val', type=int, default=100, help="number of episodes for val")
    parser.add_argument('--episodes-test', type=int, default=1000, help="number of episodes for test")

    # loss weights
    parser.add_argument('--lambda_cls', default=0., type=float)

    # other setup
    parser.add_argument('--resume', type=str, default=None, help="metamodel checkpoint to resume")
    parser.add_argument('--resume-optimizer', type=str, default=None, help="optimizer checkpoint to resume")
    parser.add_argument('--saveroot', default="./experiments/", help='Root directory to make the output directory')
    parser.add_argument('--saveprefix', default="log", help='prefix to append to the name of log directory')
    parser.add_argument('--saveargs',
                        default=["dataset", "nway", "nshot", "classifier", "backbone"]
                        , nargs='+', help='args to append to the name of log directory')
    return parser.parse_args()

def main(args):
    print(args)

    # setup dataset and dataloaders
    dataset_dict = setup_dataset(args)
    dataloader_dict = setup_dataloader(args, dataset_dict)

    # CE loss
    criterion_cls = torch.nn.CrossEntropyLoss()

    # create model
    model = ContrastResNet(args)

   # contrast loss
    criterion_contrast = ContrastiveLoss(temperature=args.temperature)

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.step_facter)

    # main training
    for epoch in range(args.epochs):
        print("epoch: %d --start from 0 and end at %d" % (epoch, args.epochs - 1))
        lr_scheduler.step()

        time1 = time.time()
        train_loss = train_one_epoch(args, dataloader_dict["train"], model, criterion_cls, criterion_contrast, optimizer)
        time2 = time.time()

        print('epoch: {}, total time: {:.2f}, train loss: {:.3f}'.format(epoch, time2 - time1, train_loss))

def train_one_epoch(args, dataloader, model, criterion_cls, criterion_contrast, optimizer):
    model.train()  # Set model to training mode

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, loss_ce, loss_con = AverageMeter(), AverageMeter(), AverageMeter()

    nway = dataloader.batch_sampler.n_way
    nshot = dataloader.batch_sampler.n_shot
    nquery = dataloader.batch_sampler.n_query

    end = time.time()

    # training lab
    for i, data in enumerate(tqdm(dataloader)):
        data_time.update(time.time() - end)

        inputs = data["input"].to(device)
        labels = data["label"].to(device)

        # ===================forward=====================
        outputs, spatial_f, global_f, avg_pool_feat = model(inputs)

        # ===================Losses=====================
        # standard CE loss
        loss_cls = criterion_cls(outputs, labels)

        # compute contrastive loss
        loss_contrast = criterion_contrast(global_f, labels=labels)

        # compute the total loss
        loss = loss_contrast * opt.lambda_global +  opt.lambda_cls * loss_cls

        # update the losses
        losses.update(loss.item())
        loss_glo.update(loss_contrast_global.item())
        loss_spa.update(loss_contrast_spatial.item())
        loss_ce.update(loss_cls.item())

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(losses.avg)


if __name__ == '__main__':
    args = setup_args()
    main(args)



