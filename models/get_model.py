import torch
import torch.optim as optim
from models.bert import BertRegressor, BertClassifier


def get_model(args):
    if args.model_name == 'BertRegressor':
        model = BertRegressor(args)
        
    elif args.model_name == 'BertClassifier':
        model = BertClassifier(args)
    else:
        raise NotImplementedError()

    optimizer = optim.Adam(model.parameters(), lr=args.training.lr)
    return model, optimizer, model.config
