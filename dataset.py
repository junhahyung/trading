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


# AM/PM model
class TradingDatasetAP(Dataset): 
    def __init__(self, data_config, mode='train'):
        self.root = data_config.root
        self.tables = data_config.tables
        self.data_config = data_config
        self.mode = mode
        self.target_dim = len(data_config.target_equity)
        self.aug_lambda = data_config.training.aug_lambda
        self.nway = data_config.training.nway
        self.use_ampm = data_config.use_ampm # use AM/PM
        if not self.use_ampm:
            raise NotImplementedError

        self.am = data_config.AM
        self.pm = data_config.PM
        self.am_index = data_config.AM_Index
        self.pm_index = data_config.PM_Index

        self.load_pd()
        self.clean_data()



        if len(data_config.test_split) == 2:
            start = data_config.test_split[0]
            end = data_config.test_split[1]
            if self.use_ampm:
                start = start*2
                end = end*2

            self.test_split = [self.tbs.iloc[start:end+1]]
            self.train_split = [self.tbs.iloc[:start], self.tbs.iloc[end+1:]]

        elif len(data_config.test_split) == 1:
            start = data_config.test_split[0]
            if self.use_ampm:
                start = start*2
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
        self.tbs.index = date_list


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


        # AM/PM generation
        self.len = len(self.tbs)
        self.tbs = pd.concat([self.tbs, self.tbs])
        self.tbs = self.tbs.sort_index()
        self.am_dict = {}
        self.pm_dict = {}
        for idx, c in enumerate(self.tbs.columns):
            if 'Equity' in c:
                # check if the equity belongs to am
                for amkey in self.am:
                    if amkey + ' Equity' in c:
                        self.am_dict[c] = idx
                        break

                for pmkey in self.pm:
                    if pmkey + ' Equity' in c:
                        self.pm_dict[c] = idx
                        break

                if c in self.am_dict and c in self.pm_dict:
                    print(c)
                    raise ValueError('wrong am pm - all')
                if c not in self.am_dict and c not in self.pm_dict:
                    print(c)
                    raise ValueError('wrong am pm - none')

            elif 'Index' in c:
                for amikey in self.am_index:
                    if amikey + ' Index' in c:
                        self.am_dict[c] = idx
                        break
                for pmikey in self.pm_index:
                    if pmikey + ' Index' in c:
                        self.pm_dict[c] = idx
                        break

                if c in self.am_dict and c in self.pm_dict:
                    print(c)
                    raise ValueError('wrong am pm - all')
                if c not in self.am_dict and c not in self.pm_dict:
                    print(c)
                    raise ValueError('wrong am pm - none')
            else:
                pass


        # change AM PM values
        for c in self.tbs.columns:
            if c in self.pm_dict:
                vals = self.tbs[c].values
                vals[0] = -0.5
                vals[2::2] = vals[1::2][:-1]
                self.tbs[c] = vals

        dates = [str(d) for d in self.tbs.index.values]
        _dates = []
        ampm = []
        for idx, d in enumerate(dates):
            if idx % 2 == 0:
                date = int(d+'0')
                ampm.append(-1)
            else:
                date = int(d+'1')
                ampm.append(1)
            _dates.append(date)
        dates = _dates
        self.tbs.index = dates

        self.tbs['ampm'] = ampm 


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
        self.target_equity_last = [n + ' ' + 'Equity_LAST_PRICE' for n in self.data_config.target_equity] # last price headers name
        self.currency_last = ['USDKRW Curncy_LAST_PRICE', 'USDJPY Curncy_LAST_PRICE', 'USDCNY Curncy_LAST_PRICE','USDEUR Curncy_LAST_PRICE']
        self.target_equity_rsi = [n + ' ' + 'Equity_RSI_14D' for n in self.data_config.target_equity] 

        # ampm mask
        am_mask = []
        for t in self.target_equity:
            is_am = False
            for amkey in self.am:
                if amkey + ' Equity' in t:
                    is_am = True
            if is_am:
                am_mask.append(1)
            else:
                am_mask.append(0)

        self.am_mask = np.array(am_mask)
        self.pm_mask = np.ones(len(self.target_equity)) - am_mask


        data_splits_y = []
        last_prices = []
        currencies = []
        rsis = []
        
        for data_split in data_splits:
            data_splits_y.append(data_split[self.target_equity])
            last_prices.append(data_split[self.target_equity_last])
            currencies.append(data_split[self.currency_last])
            rsis.append(data_split[self.target_equity_rsi])


        #print(data_splits_y[0].iloc[10:12].index.values)
        #print(self.kanji_date_to_num(data_splits_y[0].iloc[10:12].index.values))

        interval, minval = [], []
        for _t in self.target_equity:
            interval.append(self.scale_dict[_t][0])
            minval.append(self.scale_dict[_t][1])
        interval = np.array(interval)
        minval = np.array(minval)

        interval_last, minval_last = [], []
        for _t in self.target_equity_last:
            interval_last.append(self.scale_dict[_t][0])
            minval_last.append(self.scale_dict[_t][1])
        interval_last = np.array(interval_last)
        minval_last = np.array(minval_last)

        interval_cur, minval_cur = [], []
        for _t in self.currency_last:
            interval_cur.append(self.scale_dict[_t][0])
            minval_cur.append(self.scale_dict[_t][1])
        interval_cur = np.array(interval_cur)
        minval_cur = np.array(minval_cur)

        interval_rsi, minval_rsi = [], []
        for _t in self.target_equity_rsi:
            interval_rsi.append(self.scale_dict[_t][0])
            minval_rsi.append(self.scale_dict[_t][1])
        interval_rsi = np.array(interval_rsi)
        minval_rsi = np.array(minval_rsi)


        max_ntarget = max(self.data_config.ntarget)
        nhist = self.data_config.nhist
        if self.use_ampm:
            nhist = nhist*2

        self.x = []
        self.x_date = []
        self.y = []
        self.y_date = []
        self.y_origin = []
        self.y_last_origin = []
        self.y_prev_origin = []
        self.y_prev_last_origin = []
        self.y_class = []
        self.anchor = []
        self.y_curncy_prev_origin = []
        self.y_curncy_origin = []
        self.y_rsi_origin = []

        for idx, data_split in enumerate(data_splits):
            for i in range(len(data_split)-nhist-max_ntarget+1):
                _x_row = data_split.iloc[i:i+nhist]
                self.x.append(_x_row.to_numpy())
                self.x_date.append(_x_row.index.values)

                _prev = data_splits_y[idx].iloc[i+nhist-1].to_numpy() # VWAP of the last day of x
                prev_origin = self.recover_price(_prev, interval, minval)

                _prev_last = last_prices[idx].iloc[i+nhist-1].to_numpy() # closing price of the last day of x
                prev_last_origin = self.recover_price(_prev_last, interval_last, minval_last)

                prev_curncy = currencies[idx].iloc[i+nhist-1].to_numpy()
                prev_curncy_origin = self.recover_price(prev_curncy, interval_cur, minval_cur)

                _y = []
                _y_date = []
                _y_origin = []
                _y_last_origin = []
                _y_class = []
                _y_curncy_origin = []
                _y_rsi_origin = []


                for t in self.data_config.ntarget:
                    target = data_splits_y[idx].iloc[i+nhist+t-1]
                    _y.append(target.to_numpy())
                    _y_date.append([target.name][0])

                    target_origin = self.recover_price(target.to_numpy(), interval, minval)
                    target_last_origin = self.recover_price(last_prices[idx].iloc[i+nhist+t-1].to_numpy(), interval_last, minval_last)
                    target_curncy_origin = self.recover_price(currencies[idx].iloc[i+nhist+t-1].to_numpy(), interval_cur, minval_cur)
                    target_rsi_origin = self.recover_price(rsis[idx].iloc[i+nhist+t-1].to_numpy(), interval_rsi, minval_rsi)

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
                    _y_last_origin.append(target_last_origin)
                    _y_curncy_origin.append(target_curncy_origin)
                    _y_rsi_origin.append(target_rsi_origin)

                _y = np.array(_y)
                _y_class = np.array(_y_class)
                _y_origin = np.array(_y_origin)
                _y_last_origin = np.array(_y_last_origin)
                _y_curncy_origin = np.array(_y_curncy_origin)
                _y_rsi_origin = np.array(_y_rsi_origin)
                self.y.append(_y)
                self.y_date.append(_y_date)
                self.y_class.append(_y_class)
                self.y_origin.append(_y_origin)
                self.y_last_origin.append(_y_last_origin)
                self.y_prev_origin.append(prev_origin)
                self.y_prev_last_origin.append(prev_last_origin)
                self.y_curncy_prev_origin.append(prev_curncy_origin)
                self.y_curncy_origin.append(_y_curncy_origin)
                self.y_rsi_origin.append(_y_rsi_origin)


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
        self.y_last_origin = np.array(self.y_last_origin)
        self.y_prev_origin = np.array(self.y_prev_origin)
        self.y_prev_last_origin = np.array(self.y_prev_last_origin)
        self.y_curncy_prev_origin = np.array(self.y_curncy_prev_origin)
        self.y_curncy_origin = np.array(self.y_curncy_origin)
        self.y_rsi_origin = np.array(self.y_rsi_origin)
        self.anchor = np.array(self.anchor)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict['x_date'] = self.x_date[idx]
        ret_dict['y_date'] = self.y_date[idx]
        ret_dict['y_origin'] = self.y_origin[idx]
        ret_dict['y_last_origin'] = self.y_last_origin[idx]
        ret_dict['y_prev_origin'] = self.y_prev_origin[idx]
        ret_dict['y_prev_last_origin'] = self.y_prev_last_origin[idx]
        ret_dict['y_curncy_prev_origin'] = self.y_curncy_prev_origin[idx]
        ret_dict['y_curncy_origin'] = self.y_curncy_origin[idx]
        ret_dict['y_rsi_origin'] = self.y_rsi_origin[idx]
        if self.y_date[idx][0] % 2 == 0:
            ret_dict['y_mask'] = torch.LongTensor(self.am_mask)
        else:
            ret_dict['y_mask'] = torch.LongTensor(self.pm_mask)

        if self.aug_lambda != 0 and self.mode=='train':
            x = torch.FloatTensor(self.x[idx])
            return self.aug_lambda*torch.randn(x.shape) + x, torch.FloatTensor(self.y[idx]), torch.LongTensor(self.y_class[idx]), torch.FloatTensor(self.anchor[idx]), ret_dict
        else:
            return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx]), torch.LongTensor(self.y_class[idx]), torch.FloatTensor(self.anchor[idx]), ret_dict


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

'''
#--- testing! ----
import yaml
from torch.utils.data import DataLoader

with open('./config_classifier_ampm.yaml', 'r') as fp:
    data_config = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))
    
tdset = TradingDatasetAP(data_config, mode='train')
dl = DataLoader(tdset, batch_size=1, shuffle=False)
check = torch.zeros(11, dtype=torch.int8)
for idx, b in enumerate(dl):
    x, y_r, y, _, etc = b
    
    no = y_r.squeeze() != -0.5
    _check = torch.bitwise_or(check, no)
    if torch.any(_check != check):
        print(f'changed : {_check}')
        print(etc['y_date'])
        check = _check
data_config.test_split = [4875]
tdset = TradingDatasetAP(data_config, mode='test')
dl = DataLoader(tdset, batch_size=1, shuffle=False)
for idx, b in enumerate(dl):
    if idx == 0:
        x, _, y, _, etc = b
        print(etc['y_date'])
        print(idx)
    if idx == len(dl)-1:
        x, _, y, _, etc = b
        print(etc['y_date'])
        print(idx)


'''
