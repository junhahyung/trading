import json
import pickle
import tqdm
import copy

all_dict = {}

with open('filtered/from_2007/processed_dict', 'rb') as fp:
    data_dict = pickle.load(fp)

with open('filtered/from_2007/about.json') as fp:
    about_json = json.load(fp)

ordered_keys = sorted(list(about_json.items()), key=lambda x: x[1][0])
ordered_keys = [x[0] for x in ordered_keys]

reg = ['09', '10', '11', '12', '13', '14', '15', '16']
cnt = 0

all_dates = set([]) 

for ticker in data_dict.keys():
    all_dates.update(set(data_dict[ticker].keys()))

all_dates = sorted(all_dates)

for ticker in tqdm.tqdm(data_dict.keys()):
    cnt += 1
    ticker_dates = list(data_dict[ticker].keys())
    ticker_dates = sorted(ticker_dates)
    _ticker_dict = {}
    ticker_dict = {}

    prev = data_dict[ticker][ticker_dates[0]]
    #not_included = []
    for all_d in all_dates: 
        if all_d in ticker_dates:
            prev = data_dict[ticker][all_d]
            _ticker_dict[all_d] = prev
        else:
            #not_included.append(all_d)
            _ticker_dict[all_d] = prev

    for _d in _ticker_dict.keys():
        if _d[-8:-6] in reg:
            ticker_dict[_d] = _ticker_dict[_d]
            assert len(ticker_dict[_d]) == 12
        
    all_dict[ticker] = copy.deepcopy(ticker_dict)
    assert len(ticker_dict) == 30214

assert cnt == len(list(about_json.keys()))

return_dict = {}

for j, ticker in enumerate(ordered_keys):
    ticker_dict = all_dict[ticker]
    for i, date in enumerate(ticker_dict.keys()):
        if i == 0:
            firstdate = date
        if date in return_dict:
            return_dict[date] += ticker_dict[date] 
            assert len(ticker_dict[date]) == 12
        else:
            return_dict[date] = ticker_dict[date].copy()
            assert j==0
            assert len(return_dict[date]) == 12


with open('filtered/from_2007/final_processed', 'wb') as fp:
    pickle.dump(return_dict, fp)
print('saved final_processed')

'''
print(not_included)
#print('===`')
for nd in not_included[-30:]:
    print(nd)
    print(ticker_dict[nd])
'''
