import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from attrdict import AttrDict
from trainer_classifier import Trainer
from dataset import TradingDataset, TradingDatasetAP
from models.get_model import get_model


# main loop
def run(opt):
    with open('./config_classifier_ampm.yaml', 'r') as fp:
        args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

    # prepare arguments
    args.name = opt.name
    args.nhist = opt.nhist
    args.ntarget = list(opt.ntarget)
    args.use_ampm = bool(args.use_ampm)
    for i in args.ntarget:
        assert isinstance(i, int)

    print('==== N target====')
    print(args.ntarget)
    # arguments that we want to control during training
    args['training']['num_attention_heads'] = opt.num_attention_heads
    args['training']['hidden_size'] = opt.hidden_size
    args['training']['bert_layers'] = opt.bert_layers

    device = 'cuda:0'
    models = []
    optimizers = []

    # number of ensemble
    for i in range(args.training.n_ensemble):
        model, optimizer, bert_config = get_model(args)
        models.append(model)
        optimizers.append(optimizer)

    print(bert_config)

    # prepare dataset
    if args.use_ampm:
        print(args.use_ampm)
        print('~~~~')
        dataset_train = TradingDatasetAP(args, mode='train')
        dataset_test = TradingDatasetAP(args, mode='test')
    else:
        dataset_train = TradingDataset(args, mode='train')
        dataset_test = TradingDataset(args, mode='test')

    dataloader_train = DataLoader(dataset_train, batch_size=args.training.batch_size, shuffle=True, num_workers=16)
    dataloader_test = DataLoader(dataset_test, batch_size=args.training.batch_size, shuffle=False, num_workers=16)

    # prepare loss function
    loss_fn = nn.CrossEntropyLoss()

    # load trainer
    trainer = Trainer(args,
                    models,
                    optimizers,
                    dataloader_train,
                    dataloader_test,
                    loss_fn,
                    device)

    # start training
    trainer.train()

    print(f'best acc: {trainer.best_acc}')
    print(f'best confusion: {trainer.best_confusion}')
    


# parse input arguments
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default_run')
    parser.add_argument("--nhist", default=10, type=int)
    parser.add_argument("--ntarget", default=[1], type=int, nargs='+')
    parser.add_argument("--bert_layers", default=2, type=int)
    parser.add_argument("--num_attention_heads", default=6, type=int)
    parser.add_argument("--hidden_size", default=192, type=int)
    opt = parser.parse_args()

    return opt

# run training
opt = parse()
run(opt)
