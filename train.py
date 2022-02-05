import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from attrdict import AttrDict
from trainer import Trainer
from dataset import TradingDatasetFR
from models.get_model import get_model


# main loop
def run(opt):
    with open('./config/config_fr.yaml', 'r') as fp:
        args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

    # prepare arguments
    args.name = opt.name
    args.nhist = opt.nhist
    args.ntarget = opt.ntarget

    print('==== N target====')
    print(args.ntarget)
    # arguments that we want to control during training
    args['model']['num_attention_heads'] = opt.num_attention_heads
    args['model']['hidden_size'] = opt.hidden_size
    args['model']['bert_layers'] = opt.bert_layers

    device = 'cuda:0'
    models = []
    optimizers = []

    # prepare dataset
    dataset_train = TradingDatasetFR(args, mode='train')
    print(f'len train data: {len(dataset_train)}')
    dataset_test = TradingDatasetFR(args, mode='test')
    print(f'len test data: {len(dataset_test)}')

    assert dataset_train.ticker_num == dataset_test.ticker_num
    args.ticker_num = dataset_train.ticker_num
    args.feature_dim = dataset_train.feature_dim

    dataloader_train = DataLoader(dataset_train, batch_size=args.training.batch_size, shuffle=True, num_workers=16)
    dataloader_test = DataLoader(dataset_test, batch_size=args.training.batch_size, shuffle=False, num_workers=16)

    # number of ensemble
    for i in range(args.model.n_ensemble):
        model, optimizer, bert_config = get_model(args)
        models.append(model)
        optimizers.append(optimizer)

    print(bert_config)

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
    parser.add_argument("--nhist", default=30, type=int)
    parser.add_argument("--ntarget", default=1, type=int)
    parser.add_argument("--bert_layers", default=2, type=int)
    parser.add_argument("--num_attention_heads", default=12, type=int)
    parser.add_argument("--hidden_size", default=2304, type=int)
    opt = parser.parse_args()

    return opt

# run training
opt = parse()
run(opt)
