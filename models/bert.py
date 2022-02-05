import torch
import torch.nn as nn
from attrdict import AttrDict
from transformers import BertModel, BertConfig


'''
class BertRegressor(nn.Module):
    def __init__(self, args):
       super(BertRegressor, self).__init__()
       bert_config = BertConfig()

       bert_config.num_hidden_layers = args.training.bert_layers
       bert_config.hidden_dropout_prob = args.training.dropout
       bert_config.attention_probs_dropout_prob = args.training.attention_dropout
       bert_config.num_attention_heads = args.training.num_attention_heads
       bert_config.hidden_size = args.training.hidden_size
       bert_config.intermediate_size = args.training.intermediate_size
       self.config = bert_config

       #print(bert_config)
       self.bert = BertModel(bert_config)
       classes = len(args.target_equity) * len(args.ntarget)
       if args.use_ampm:
           self.input_dim = args.input_dim + 2 # num features + dates
       else:
           self.input_dim = args.input_dim + 1 # num features + dates
       self.in_conv = nn.Conv1d(self.input_dim, bert_config.hidden_size, kernel_size=1)
       self.out_linear = nn.Linear(self.bert.config.hidden_size, classes)
       self.final_act = nn.Tanh()


    def forward(self, inputs_embeds, prev):
       # inputs_embeds: (batch_size, seq_length, input_dim)
       out = torch.transpose(inputs_embeds, 2, 1)
       out = torch.transpose(self.in_conv(out), 2, 1) # (batch_size, seq_length, hidden_size)
       _, out = self.bert(inputs_embeds=out, return_dict=False)
       out = self.final_act(self.out_linear(out))
       out = out + prev.view(prev.shape[0], -1)
       return out
'''


class BertClassifier(nn.Module):
    def __init__(self, args):
       super(BertClassifier, self).__init__()
       bert_config = BertConfig()

       bert_config.num_hidden_layers = args.model.bert_layers
       bert_config.hidden_dropout_prob = args.model.dropout
       bert_config.attention_probs_dropout_prob = args.model.attention_dropout
       bert_config.num_attention_heads = args.model.num_attention_heads
       bert_config.hidden_size = args.model.hidden_size
       bert_config.intermediate_size = args.model.intermediate_size
       self.config = bert_config

       #print(bert_config)
       self.bert = BertModel(bert_config)
       self.classes = args.ticker_num
       self.feature_dim = args.feature_dim
       self.input_dim = args.feature_dim * self.classes
       self.in_conv = nn.Conv1d(self.input_dim, bert_config.hidden_size, kernel_size=1)
       self.out_linear = nn.Linear(self.bert.config.hidden_size, self.classes)
       self.final_layers = nn.Sequential(nn.Linear(self.classes, self.classes*2), nn.ReLU(), nn.Linear(self.classes*2, self.classes*2))


    def forward(self, inputs_embeds):
       # inputs_embeds: (batch_size, seq_length, input_dim)
       price_ind = self.feature_dim * torch.arange(self.classes) + 3
       price = inputs_embeds[:,-1, price_ind] # (batch_size, classes)
       out = torch.transpose(inputs_embeds, 2, 1)
       out = torch.transpose(self.in_conv(out), 2, 1) # (batch_size, seq_length, hidden_size)
       _, out = self.bert(inputs_embeds=out, return_dict=False)
       out = self.out_linear(out)
       out = out + price
       out = self.final_layers(out)
       return out

'''
import yaml
with open('./data_config.yaml', 'r') as fp:
    args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

device = 'cuda:5'
bert = Bert(args, 314).to(device)
inputs_embeds = torch.randn(1,5,314).to(device)
out = bert(inputs_embeds)
print(out)
'''
