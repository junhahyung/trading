import os
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
    def __init__(self, data_config):
        self.root = data_config.root
        self.tables = data_config.tables

        self.load_pd()
        print(self.tbs)
        print(self.tbs[0].columns)
        print(self.tbs[1].columns)

    def load_pd(self):
        self.tbs = []
        for table_key in self.tables.keys():
            table_conf = AttrDict(self.tables[table_key])
            
            csv_dir = os.path.join(self.root, table_conf.dir) 
            tb = pd.read_csv(csv_dir, header=table_conf.subheader, index_col=table_conf.index_col)


            headings = pd.read_csv(csv_dir, skiprows=3, nrows=1)
            headings = [head for head in headings.columns if 'Unnamed' not in head]
            print(len(headings))

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

    def __len__(self):
        return 0

    def __getitem__(self):
        return 0


#--- testing! ----

import yaml
with open('./data_config.yaml', 'r') as fp:
    data_config = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))
    
tdset = TradingDataset(data_config)
