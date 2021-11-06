import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from attrdict import AttrDict
from trainer_classifier import Trainer
from dataset import TradingDataset
from models.get_model import get_model


# main loop
def run(opt):
    with open('./config_classifier.yaml', 'r') as fp:
        args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

    args.name = opt.name
    args.nhist = opt.nhist
    args.ntarget = opt.ntarget
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
    dataset_train = TradingDataset(args, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.training.batch_size, shuffle=True, num_workers=16)
    dataset_test = TradingDataset(args, mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.training.batch_size, shuffle=False, num_workers=16)

    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(args,
                    models,
                    optimizers,
                    dataloader_train,
                    dataloader_test,
                    loss_fn,
                    device)

    trainer.train()

    print(f'best acc: {trainer.best_acc}')
    print(f'best confusion: {trainer.best_confusion}')
    


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default_run')
    parser.add_argument("--nhist", default=10, type=int)
    parser.add_argument("--ntarget", default=1, type=int, nargs='+')
    parser.add_argument("--bert_layers", default=2, type=int)
    parser.add_argument("--num_attention_heads", default=6, type=int)
    parser.add_argument("--hidden_size", default=192, type=int)
    opt = parser.parse_args()

    return opt

opt = parse()
run(opt)
