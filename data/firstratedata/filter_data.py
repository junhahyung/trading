# usage: python filter_data.py 2007

import os
import re
import sys
import json
import tqdm
import pickle
import numpy as np
from datetime import datetime


def main(thres):
    thres_date = datetime.strptime(thres+'0101 00:00:00', '%Y%m%d %H:%M:%S')
    thres_end_date = datetime.strptime('20220101 00:00:00', '%Y%m%d %H:%M:%S')


    h_dir = '/home/nas2_userG/junhahyung/trading/data/firstratedata/1hour/'
    #m_dir = '/home/nas2_userG/junhahyung/trading/data/firstratedata/5min'

    out_dir = '/home/nas2_userG/junhahyung/trading/data/firstratedata/filtered/'
    out_dir = os.path.join(out_dir, 'from_'+thres)
    os.makedirs(out_dir, exist_ok=True)

    h_list = os.listdir(h_dir)
    #m_list = os.listdir(m_dir)

    #assert len(h_list) == len(m_list)
    print(len(h_list))
    #print(len(m_list))

    about_json = {}
    array_collection = []

    all_array = {}

    return_dict = {}

    cnt = 0
    for ht in tqdm.tqdm(h_list):
        if ht[-4:] != '.txt':
            continue

        hdir = os.path.join(h_dir, ht)
        print(hdir)
        with open(hdir, 'r') as fp:
            lines = fp.readlines()
            start_date = lines[0].strip().split(',')[0]
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            end_date = lines[-1].strip().split(',')[0]
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        if start_date < thres_date and end_date > thres_end_date:
            array = []
            dates = []
            
            with open(hdir, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    data = line.strip().split(',')
                    date = data[0]
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    if date >= thres_date and date < thres_end_date:
                        _st = ''.join(re.findall("\d+", str(date)))
                        st = [float(_st[:8]), float(_st[4:6]), float(_st[6:8]), float(_st[8:10])] # yyyymmdd, mm, dd, hh
                        pr = [float(p) for p in [data[4], data[5]]]
                        array.append(st+pr)
                        dates.append(str(date))

                        if str(date) in all_array:
                            all_array[str(date)].append(cnt)
                        else:
                            all_array[str(date)] = [cnt]


            if len(array) < 1:
                continue

            return_dict[ht] = {}

            array = np.array(array)
            _max = np.max(array,axis=0)
            _min = np.min(array,axis=0)
            inter = _max - _min
            if not inter.all():
                print('~~~~~')
                print(_max)
                print(_min)
                print(ht)
                print(inter)
                print('inter is 0!!!')
            array = np.concatenate((array, ((array - _min)/inter*2 - 1)), axis=1)
            assert len(array) == len(dates)
            for i, d in enumerate(dates):
                #print(array[i].tolist())
                return_dict[ht][d] = array[i].tolist()

            about_json[ht] = [cnt, _max.tolist(), _min.tolist(), inter.tolist()]
            array_collection.append(array)

            cnt += 1

    sorted_re = sorted([(d, len(c)) for d,c in all_array.items()], key=lambda x: x[0])
    with open('sorted_re','wb') as fp:
        pickle.dump(sorted_re, fp)

    i = 0

    print(f'cnt: {cnt}')
    print(sorted_re)

    with open(os.path.join(out_dir, 'processed_dict'),'wb') as fp:
        pickle.dump(return_dict, fp)

    #array_collection = np.array(array_collection)
    #np.save(os.path.join(out_dir, 'data.npy'), array_collection)
    with open(os.path.join(out_dir, 'about.json'), 'w') as f:
        json.dump(about_json, f)



if __name__ == '__main__':
    main(sys.argv[1])
