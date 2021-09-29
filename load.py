import os
from attrdict import AttrDict
import pandas as pd
from torch.utils.data import Dataset


data_dir = './data/Semicap_DailyData/Currencies_DAILY-Table 1.csv'
cur = pd.read_csv(data_dir, header=5, index_col=0)

DROP_SYM = ['PX_VOLUME']

drop_cols = []
keys = cur.keys()
for SYM in DROP_SYM:
    for key in keys:
        if SYM in key:
            drop_cols.append(key)

tb = cur.drop(drop_cols, axis=1)



class TradingDataset(Dataset): 
    def __init__(self, data_config):
        self.root = data_config.root
        self.tables = data_config.tables

        self.load_pd()
        print(self.tbs)

    def load_pd(self):
        self.tbs = []
        for table_key in self.tables.keys():
            table_conf = AttrDict(self.tables[table_key])
            
            csv_dir = os.path.join(self.root, table_conf.dir) 
            tb = pd.read_csv(csv_dir, header=table_conf.header, index_col=table_conf.index_col)

            drop_cols = []
            keys = tb.keys()
            for SYM in table_conf.drop_sym:
                for key in keys:
                    if SYM in key:
                        drop_cols.append(key)

            tb = cur.drop(drop_cols, axis=1)
            self.tbs.append(tb)

    def __len__(self):
        return 0

    def __getitem__(self):
        return 0


#--- testing! ----

import yaml
with open('./data_config.yaml', 'r') as fp:
    data_config = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))
    
tdset = TradingDataset(data_config)
