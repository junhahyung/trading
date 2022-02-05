import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from attrdict import AttrDict
from trainer_regressor import Trainer
from dataset import TradingDataset
from models.get_model import get_model


# main loop
def main():
    with open('./config.yaml', 'r') as fp:
        args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

    device = 'cuda:0'
    model, optimizer = get_model(args)
    dataset_train = TradingDataset(args, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.training.batch_size, shuffle=True, num_workers=16)
    dataset_test = TradingDataset(args, mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.training.batch_size, shuffle=False, num_workers=16)

    loss_fn = nn.MSELoss()

    trainer = Trainer(args,
                    model,
                    optimizer,
                    dataloader_train,
                    dataloader_test,
                    loss_fn,
                    device)

    trainer.train()
    

main()
