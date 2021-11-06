import os
import re
import numpy as np
import torch
from attrdict import AttrDict
import pandas as pd
from torch.utils.data import Dataset


'''
data_dir = './data/Semicap_DailyData/Daily_share_px-Table 1.csv'
cur = pd.read_csv(data_dir, header=5, index_col=0, na_values=0).fillna(0)


headings = pd.read_csv(data_dir, skiprows=3, nrows=1)
headings = [head for head in headings.columns if 'Unnamed' not in head]
print(len(headings))

eph = 7

new_header_map = {}
for idx, n in enumerate(cur.columns):
    new_header_map[n] = headings[idx%eph] + '_' +n.split('.')[0]
    
cur.rename(columns=new_header_map, inplace=True)
'''


class TradingDataset(Dataset): 
    def __init__(self, data_config, mode='train'):
        self.root = data_config.root
        self.tables = data_config.tables
        self.data_config = data_config
        self.mode = mode
        self.target_dim = len(data_config.target_equity)
        self.aug_lambda = data_config.training.aug_lambda
        self.nway = data_config.training.nway

        self.load_pd()
        self.clean_data()


        if len(data_config.test_split) == 2:
            start = data_config.test_split[0]
            end = data_config.test_split[1]

            self.test_split = [self.tbs.iloc[start:end+1]]
            self.train_split = [self.tbs.iloc[:start], self.tbs.iloc[end+1:]]

        elif len(data_config.test_split) == 1:
            start = data_config.test_split[0]
            self.test_split = [self.tbs.iloc[start:]]
            self.train_split = [self.tbs.iloc[:start]]

        else:
            raise ValueError(f"{len(data_config.test_split)} elements given for data_config.test_split")




        self.configure_dataset()

    # 1月1日2000年-> 20000101
    @staticmethod
    def kanji_date_to_num(date_list):
        ret = []
        for date in date_list:
            date = re.findall(r'\d+', date)
            date = date[2].zfill(4) + date[0].zfill(2) + date[1].zfill(2)
            ret.append(int(date))

        return ret


    def load_pd(self):
        self.tbs = []
        for table_key in self.tables.keys():
            table_conf = AttrDict(self.tables[table_key])
            
            csv_dir = os.path.join(self.root, table_conf.dir) 
            tb = pd.read_csv(csv_dir, header=table_conf.subheader, index_col=table_conf.index_col)


            headings = pd.read_csv(csv_dir, skiprows=3, nrows=1)
            headings = [head for head in headings.columns if 'Unnamed' not in head]

            new_header_map = {}
            for idx, n in enumerate(tb.columns):
                new_header_map[n] = headings[idx//table_conf.eph] + '_' +n.split('.')[0]

            tb.rename(columns=new_header_map, inplace=True)

            if 'drop_sym' in table_conf.keys():
                drop_cols = []
                keys = tb.keys()
                for SYM in table_conf.drop_sym:
                    for key in keys:
                        if SYM in key:
                            drop_cols.append(key)

                tb = tb.drop(drop_cols, axis=1)

            self.tbs.append(tb)

        self.tbs = pd.concat(self.tbs, axis=1)

        # add date as a column
        date_list = list(self.tbs.index)
        date_list = self.kanji_date_to_num(date_list)
        self.tbs['dates'] = date_list


    def clean_data(self):
        # replace na values with 0
        self.tbs = self.tbs.fillna(0)

        self.scale_dict = {}
        
        for i, col in enumerate(self.tbs.columns):
            
            maxval = max(self.tbs.iloc[:,i])
            minval = min(self.tbs.iloc[:,i])
            interval = maxval - minval
            if interval == 0.0:
                interval = 1.

            # scaling: -0.5 to 0.5
            self.tbs.iloc[:,i] = 0.5*(2.*(self.tbs.iloc[:,i]-minval) / interval - 1)

            # store scaling factors
            self.scale_dict[col] = [interval, minval]


    @staticmethod
    def recover_price(normalized, interval, minval):
        price = (2*normalized+1)*interval/2 + minval
        return price
    
    def configure_dataset(self):
        if self.mode == 'train':
            data_splits = self.train_split
        elif self.mode == 'test':
            data_splits = self.test_split
        else:
            raise ValueError(f'inappropriate dataset mode: {self.mode}')

        # target equity
        self.target_equity = [n + ' ' + self.data_config.target_subheader for n in self.data_config.target_equity]
        data_splits_y = []
        for data_split in data_splits:
            data_splits_y.append(data_split[self.target_equity])

        #print(data_splits_y[0].iloc[10:12].index.values)
        #print(self.kanji_date_to_num(data_splits_y[0].iloc[10:12].index.values))

        interval, minval = [], []
        for _t in self.target_equity:
            interval.append(self.scale_dict[_t][0])
            minval.append(self.scale_dict[_t][1])

        interval = np.array(interval)
        minval = np.array(minval)

        max_ntarget = max(self.data_config.ntarget)
        nhist = self.data_config.nhist

        self.x = []
        self.x_date = []
        self.y = []
        self.y_date = []
        self.y_origin = []
        self.y_class = []
        self.anchor = []

        for idx, data_split in enumerate(data_splits):
            for i in range(len(data_split)-nhist-max_ntarget+1):
                _x_row = data_split.iloc[i:i+nhist]
                self.x.append(_x_row.to_numpy())
                self.x_date.append(self.kanji_date_to_num(_x_row.index.values))

                _y = []
                _y_date = []
                _y_origin = []
                _y_class = []
                _prev = data_splits_y[idx].iloc[i+nhist-1].to_numpy()

                for t in self.data_config.ntarget:
                    target = data_splits_y[idx].iloc[i+nhist+t-1]
                    _y.append(target.to_numpy())
                    _y_date.append(self.kanji_date_to_num([target.name])[0])

                    target_origin = self.recover_price(target.to_numpy(), interval, minval)
                    prev_origin = self.recover_price(_prev, interval, minval)
                    earnings = np.zeros(self.target_dim)
                    for tdim in range(len(earnings)):
                        denom = prev_origin[tdim] if prev_origin[tdim] != 0 else 1.
                        earnings[tdim] = (target_origin[tdim] - prev_origin[tdim]) / denom

                    # 3way 0: neutral / 1: pos / 2: neg 
                    # 2way 0: neg / 1: pos 
                    label = np.zeros(self.target_dim)
                    if self.nway == 3:
                        pos = np.ones(self.target_dim)
                        neg = pos + pos
                        label = np.where(earnings >= 0.01, pos, label)
                        label = np.where(earnings <= -0.01, neg, label)
                        _y_class.append(label)
                    elif self.nway == 2:
                        pos = np.ones(self.target_dim)
                        label = np.where(earnings > 0, pos, label)
                        _y_class.append(label)

                    _y_origin.append(target_origin)

                _y = np.array(_y)
                _y_class = np.array(_y_class)
                _y_origin = np.array(_y_origin)
                self.y.append(_y)
                self.y_date.append(_y_date)
                self.y_class.append(_y_class)
                self.y_origin.append(_y_origin)

                # add anchor
                _anchor = []
                _prev_mean = np.mean(data_splits_y[idx].iloc[i:i+nhist].to_numpy(), axis=0)
                _anchor.append(_prev_mean)
                _anchor.append(_prev)
                _anchor = np.array(_anchor)
                self.anchor.append(_anchor)

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.y_class = np.array(self.y_class)
        self.y_origin = np.array(self.y_origin)
        self.anchor = np.array(self.anchor)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict['x_date'] = self.x_date[idx]
        ret_dict['y_date'] = self.y_date[idx]
        ret_dict['y_origin'] = self.y_origin[idx]

        if self.aug_lambda != 0 and self.mode=='train':
            x = torch.FloatTensor(self.x[idx])
            return self.aug_lambda*torch.randn(x.shape) + x, torch.FloatTensor(self.y[idx]), torch.LongTensor(self.y_class[idx]), torch.FloatTensor(self.anchor[idx]), ret_dict
        else:
            return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx]), torch.LongTensor(self.y_class[idx]), torch.FloatTensor(self.anchor[idx]), ret_dict

#--- testing! ----

'''
import yaml
with open('./config_classifier.yaml', 'r') as fp:
    data_config = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))
    
tdset = TradingDataset(data_config, mode='train')
di = tdset[0][-1]
print(di['x_date'])
print('----')
print(di['y_date'])


length=len(tdset)
zero = 0
one = 0
two = 0
for i in range(length):
    zero += torch.sum(tdset[i][2]==0)
    one += torch.sum(tdset[i][2]==1)
    two += torch.sum(tdset[i][2]==2)

print(zero)
print(one)
print(two)
print(len(tdset))
print('y')
print(tdset[0][1][0].shape)
print(tdset[0][1][0])
print('anchor')
print(tdset[0][3].shape)
print(tdset[0][3][1])
print('y_class')
print(tdset[0][2].shape)
print(tdset[0][2])
'''
