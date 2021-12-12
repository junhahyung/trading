import torch
import torch.nn as nn
from dataset import TradingDatasetAP
from models.get_model import get_model
from torch.utils.data import DataLoader
from attrdict import AttrDict
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from metric import expectancy, expectunity


class BackTester:
    def __init__(self, args, model_pth, aum, tax_rate):
        self.dataset = TradingDatasetAP(args, mode='test')
        print(f'total length: {len(self.dataset)}')
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.predictor = Predictor(args, model_pth)
        self.target_equity = args.target_equity
        self.trader = Trader_AMPM(self.predictor, args, aum, self.target_equity, tax_rate)

    def run(self, end_date=None, do_analysis=True):
        self.total_length = len(self.dataloader)
        self.strat_cal_days = 0
        end_count = 0

        for idx, data in enumerate(tqdm(self.dataloader)):
            x, _, y, _, ret_dict = data

            date = ret_dict['x_date'].tolist()[0][-1]
            vwap = ret_dict['y_prev_origin'][0].tolist()
            closing_price = ret_dict['y_prev_last_origin'][0].tolist()
            
            next_date = ret_dict['y_date'][0].item()
            next_vwap = ret_dict['y_origin'][0][0].tolist()
            next_closing_price = ret_dict['y_last_origin'][0][0].tolist()

            input_dict = {'x': x,
                          'y': y,
                          'date': date,
                          'next_date': next_date,
                          'vwap': vwap,
                          'closing_price': closing_price,
                          'ret_dict': ret_dict}

            if idx == 0:
                self.start_date = date

            # the last day
            if (end_date is not None and date >= end_date) or (idx >= self.total_length - max(args.ntarget)):
                end_count += 1
                self.trader.cleanup_trade(**input_dict)
                self.trader.update_aum(**input_dict)
                self.end_date = date
                self.strat_cal_days += 1
            else:
                self.trader.cleanup_trade(**input_dict)
                self.trader.trade(**input_dict)
                self.strat_cal_days += 1

            if end_count >= max(args.ntarget):
                break


        if do_analysis:
            self.analysis()

        return self.trader.aum


    def analysis(self):
        # plot aum graph
        # ----------------------------------------------------#
        aum_plot = self.trader.aum_plot
        
        xs = list(aum_plot.keys())
        plt.plot(xs, list(aum_plot.values()))
        plt.xticks(xs[::100], [x[2:-1] for x in xs[::100]], rotation=45)
        plt.savefig('aumplot.jpg')
        # ----------------------------------------------------#

        # account should be empty
        for t in self.target_equity:
            assert self.trader.account_long[t] == {}
            assert self.trader.account_short[t] == {}
        print('[SUCCESS] - account empty')


        # equity-wise stats
        # ----------------------------------------------------#
        sr_long, strike_long, fail_long, wins_long, losses_long = self.strike_rate(self.trader.long_trade)
        sr_short, strike_short, fail_short, wins_short, losses_short = self.strike_rate(self.trader.short_trade)

        long_earnings = self.calc_earnings(self.trader.long_trade)
        short_earnings = self.calc_earnings(self.trader.short_trade)
        total_earnings = 0

        expectancy_dict, expectunity_dict, total_dict = self.calc_expectunity(wins_long, losses_long, wins_short, losses_short, self.strat_cal_days)

        for t in self.target_equity:
            total_earnings += long_earnings[t]
            total_earnings += short_earnings[t]
            print('======')
            print(t)
            print(f'long strike rate: {sr_long[t]} ({strike_long[t]} / {strike_long[t]+fail_long[t]})')
            print(f'long earnings: {long_earnings[t]}')
            print(f'long expectancy: {expectancy_dict[t]["long"]}')
            print(f'long expectunity: {expectunity_dict[t]["long"]}')
            print(f'short strike rate: {sr_short[t]} ({strike_short[t]} / {strike_short[t]+fail_short[t]})')
            print(f'short earnings: {short_earnings[t]}')
            print(f'short expectancy: {expectancy_dict[t]["short"]}')
            print(f'short expectunity: {expectunity_dict[t]["short"]}')

            print('======')

        # ----------------------------------------------------#

        # overall stats
        # ----------------------------------------------------#
        print('===OVERALL STATS===')
        print(f'start date: {self.start_date} - end date: {self.end_date}')
        print(f'AUM: {self.trader.aum}')
        print(f'total trade count: {self.trader.trade_cnt}')
        print(f'total trade days(AMPM): {self.trader.initiate} / {self.strat_cal_days}')
        print(f'long count: {self.trader.long_cnt}')
        print(f'short count: {self.trader.short_cnt}')



        print(f'total_earnings: {total_earnings}')
        print(f'total long expectancy: {total_dict["total_long_expectancy"]}')
        print(f'total long expectunity: {total_dict["total_long_expectunity"]}')
        print(f'total short expectancy: {total_dict["total_short_expectancy"]}')
        print(f'total short expectunity: {total_dict["total_short_expectunity"]}')
        print(f'total expectancy: {total_dict["total_expectancy"]}')
        print(f'total expectunity: {total_dict["total_expectunity"]}')

        # ----------------------------------------------------#


    def calc_expectunity(self, wins_long, losses_long, wins_short, losses_short, strat_cal_days):
        expectancy_dict = {}
        expectunity_dict = {}
        total_wins = []
        total_long_wins = []
        total_short_wins = []
        total_losses = []
        total_long_losses = []
        total_short_losses = []

        for t in self.target_equity:
            expectancy_dict[t] = {}
            expectancy_dict[t]['total'] = expectancy(wins_long[t] + wins_short[t], losses_long[t] + losses_short[t])
            expectancy_dict[t]['long'] = expectancy(wins_long[t], losses_long[t])
            expectancy_dict[t]['short'] = expectancy(wins_short[t], losses_short[t])

            expectunity_dict[t] = {}
            expectunity_dict[t]['total'] = expectunity(wins_long[t] + wins_short[t], losses_long[t] + losses_short[t], strat_cal_days)
            expectunity_dict[t]['long'] = expectunity(wins_long[t], losses_long[t], strat_cal_days)
            expectunity_dict[t]['short'] = expectunity(wins_short[t], losses_short[t], strat_cal_days)

            total_wins += wins_long[t] + wins_short[t]
            total_long_wins += wins_long[t]
            total_short_wins += wins_short[t]

            total_losses += losses_long[t] + losses_short[t]
            total_long_losses += losses_long[t]
            total_short_losses += losses_short[t]

        total_dict = {}
        total_dict['total_expectancy'] = expectancy(total_wins, total_losses)
        total_dict['total_long_expectancy'] = expectancy(total_long_wins, total_long_losses)
        total_dict['total_short_expectancy'] = expectancy(total_short_wins, total_short_losses)

        total_dict['total_expectunity'] = expectunity(total_wins, total_losses, strat_cal_days)
        total_dict['total_long_expectunity'] = expectunity(total_long_wins, total_long_losses, strat_cal_days)
        total_dict['total_short_expectunity'] = expectunity(total_short_wins, total_short_losses, strat_cal_days)

        return expectancy_dict, expectunity_dict, total_dict


    def calc_earnings(self, trade):
        earnings = {}

        for t in self.target_equity:
            _earnings = 0
            _trade = trade[t]
            for date in _trade.keys():
                for elem in _trade[date]:
                    if isinstance(elem, float):
                        assert elem > 0
                        _earnings += elem
                    elif isinstance(elem, list):
                        assert elem[0] < 0
                        _earnings += elem[0]
                    else:
                        raise ValueError
            earnings[t] = _earnings
        return earnings


    def strike_rate(self, trade):
        strike_rate = {}
        strike = {}
        fail = {}

        wins = {}
        losses = {}
        for t in self.target_equity:
            strike[t] = 0
            fail[t] = 0
            wins[t] = []
            losses[t] = []
            _trade = trade[t]
            for date in _trade.keys():
                for elem in _trade[date]:
                    if isinstance(elem, list):
                        p, ndate = elem
                        assert p < 0
                        for nelem in _trade[ndate]:
                            if isinstance(nelem, float):
                                assert nelem > 0
                                if p+nelem > 0:
                                    strike[t] += 1
                                    wins[t].append(p+nelem)
                                else:
                                    fail[t] += 1
                                    losses[t].append(p+nelem)
                                break
                    elif isinstance(elem, float):
                        assert elem > 0

        for t in self.target_equity:
            _s = strike[t]
            _f = fail[t]
            if _s + _f > 0:
                strike_rate[t] = _s / (_s+_f)
            else:
                strike_rate[t] = -1
        return strike_rate, strike, fail, wins, losses
            


class Trader_AMPM:
    def __init__(self, predictor, args, aum, target_equity, tax_rate):
        self.predictor = predictor
        self.args = args
        self.aum = aum
        self.aum_plot = {}
        self.cash = aum
        self.target_equity = target_equity
        self.trade_cnt = 0
        self.initiate = 0
        self.long_cnt = 0
        self.short_cnt = 0
        #self.long_earnings = 0
        #self.short_earnings = 0
        self.long_trade = {}
        self.short_trade = {}
        self.account_long = {}
        self.account_short = {}
        self.schedule = {}
        self.tax_rate = tax_rate
        for t in self.target_equity:
            self.account_long[t] = {}
            self.account_short[t] = {}
            self.schedule[t] = {}
            self.long_trade[t] = {}
            self.short_trade[t] = {} 


    def log(self,date, account_long, account_short, cash, aum):
        print('==========')
        print(f'[{date}] Cash: {cash}  AUM: {aum}')
        '''
        print('--account_long--')
        for t in account_long.keys():
            for p in account_long[t].keys():
                print(f'{t} : {account_long[t][p]}')
        print('--account_short--')
        for t in account_short.keys():
            for p in account_short[t].keys():
                print(f'{t} : {account_short[t][p]}')
        '''
        print('==========')


    def cleanup_trade(self, **input_dict):
        date = input_dict['date']
        vwap = input_dict['vwap']
        ret_dict = input_dict['ret_dict']
        er = ret_dict['y_curncy_prev_origin'][0].tolist()

        for idx, equity_name in enumerate(self.target_equity):
            _vwap = vwap[idx]
            _vwap_usd  = self.to_usd(equity_name, _vwap, er)

            s_dates = list(self.schedule[equity_name].keys())
            for s_date in s_dates:
                if date == s_date:
                    amount = self.schedule[equity_name][date]
                    del self.schedule[equity_name][date]

                    # sell
                    if amount < 0:
                        sell = -1*amount
                        _total = _vwap_usd*sell
                        _tax = _total*self.tax_rate
                        #self.cash += _vwap_usd*sell 
                        self.cash += (_total - _tax)
                        self.long_trade[equity_name][date] = [_total - _tax]
                        ps = list(self.account_long[equity_name].items())
                        for p, _amount in ps:
                            _left = max(0, _amount - sell)
                            if _left == 0:
                                del self.account_long[equity_name][p]
                                sell -= _amount
                            else:
                                self.account_long[equity_name][p] = _left
                                sell = 0
                            if sell == 0:
                                break

                    # short cover
                    elif amount > 0:
                        cover = amount
                        ps = list(self.account_short[equity_name].items())
                        for p, _amount in ps:
                            _left = max(0, _amount - cover)
                            if _left == 0:
                                del self.account_short[equity_name][p]
                                cover -= _amount
                            else:
                                self.account_short[equity_name][p] = _left
                                cover = 0
                            _total = (2*p - _vwap_usd)*(_amount - _left)
                            _tax = _total * self.tax_rate
                            #self.cash += (2*p - _vwap_usd)*(_amount - _left) 
                            self.cash += (_total - _tax)
                            self.short_trade[equity_name][date] = [_total - _tax] 
                            if cover == 0:
                                break
                    else:
                        raise NotImplementedError


    def trade(self, **input_dict):
        x = input_dict['x']
        y = input_dict['y']
        date = input_dict['date']
        next_date = input_dict['next_date']
        vwap = input_dict['vwap']
        closing_price = input_dict['closing_price']
        ret_dict = input_dict['ret_dict']

        pred = self.predictor.predict(x)
        confidence, max_ind = torch.max(pred, -1)

        y_mask = ret_dict['y_mask']
        y_mask = y_mask.view(-1)
        y_mask_ind = y_mask.nonzero().view(-1).tolist()

        # exchange rate
        er = ret_dict['y_curncy_prev_origin'][0].tolist()

        # rsi
        rsi = ret_dict['y_rsi_origin'][0][0].tolist()

        # how many equities to trade
        equity_num = 0
        # bootstap run to decide trade amount
        for idx, equity_name in enumerate(self.target_equity):
            if idx in y_mask_ind:
                #equity_name = self.target_equity[idx]
                _confidence = confidence[idx]
                _max_ind = max_ind[idx]
                _vwap = vwap[idx]
                _closing_price = closing_price[idx]
                _rsi = rsi[idx]

                _long, _short = self.decision_algo(equity_name, _max_ind, _confidence, _vwap, _closing_price, _rsi)
                if _long or _short:
                    self.trade_cnt += 1
                    equity_num += 1
                if _long:
                    self.long_cnt += 1
                elif _short:
                    self.short_cnt += 1

        if equity_num == 0:
            self.aum_plot[str(date)] = self.aum
            return
        else:
            self.initiate += 1

        for idx, equity_name in enumerate(self.target_equity):
            log = False
            if idx in y_mask_ind:
                _confidence = confidence[idx]
                _max_ind = max_ind[idx]
                _vwap = vwap[idx]
                _closing_price = closing_price[idx]
                _rsi = rsi[idx]

                _long, _short = self.decision_algo(equity_name, _max_ind, _confidence, _vwap, _closing_price, _rsi)
                _closing_price_usd = self.to_usd(equity_name, _closing_price, er)


                #수량결정
                budget = (self.cash*0.99) / equity_num
                amount = budget // _closing_price_usd
                if amount == 0:
                    '''
                    print('~~~~')
                    print(self.cash)
                    print(budget)
                    print(_closing_price_usd)
                    '''
                    continue

                if _long:
                    log = True
                    if _closing_price_usd in self.account_long[equity_name]:
                        self.account_long[equity_name][_closing_price_usd] += amount
                    else:
                        self.account_long[equity_name][_closing_price_usd] = amount
                    _total = _closing_price_usd*amount
                    _tax = _total*self.tax_rate
                    self.cash -= (_total + _tax)

                    if date in self.long_trade[equity_name]:
                        self.long_trade[equity_name][date].append([-1*(_total + _tax), next_date])
                    else:
                        self.long_trade[equity_name][date] = [[-1*(_total + _tax), next_date]]

                    if next_date not in self.schedule[equity_name]:
                        self.schedule[equity_name][next_date] = -1*amount
                    else:
                        raise NotImplementedError
                        self.schedule[equity_name][next_date] += -1*amount
                elif _short:
                    log = True
                    if _closing_price_usd in self.account_short[equity_name]:
                        self.account_short[equity_name][_closing_price_usd] += amount
                    else:
                        self.account_short[equity_name][_closing_price_usd] = amount
                    _total = _closing_price_usd*amount
                    _tax = _total*self.tax_rate
                    self.cash -= (_total + _tax)

                    if date in self.short_trade[equity_name]:
                        self.short_trade[equity_name][date].append([-1*(_total + _tax), next_date])
                    else:
                        self.short_trade[equity_name][date] = [[-1*(_total + _tax), next_date]]

                    if next_date not in self.schedule[equity_name]:
                        self.schedule[equity_name][next_date] = amount
                    else:
                        raise NotImplementedError
                        self.schedule[equity_name][next_date] += amount
                else:
                    pass


        '''
        # update aum
        aum = 0
        for idx, equity_name in enumerate(self.target_equity):
            _closing_price = closing_price[idx]
            _closing_price_usd = self.to_usd(equity_name, _closing_price, er)
            for _, amount in self.account_long[equity_name].items():
                aum += _closing_price_usd*amount

            for p, amount in self.account_short[equity_name].items():
                aum += (2*p-_closing_price_usd)*amount

        aum += self.cash
        self.aum = aum
        self.aum_plot[str(date)] = self.aum
        '''
        self.update_aum(**input_dict)

        '''
        if log:
            self.log(date, self.account_long, self.account_short, self.cash, self.aum)
            #self.log(date, None, None, self.cash, self.aum)
        '''

    def update_aum(self, **input_dict):
        closing_price = input_dict['closing_price']
        er = input_dict['ret_dict']['y_curncy_prev_origin'][0].tolist()
        date = input_dict['date']
        # update aum
        aum = 0
        for idx, equity_name in enumerate(self.target_equity):
            _closing_price = closing_price[idx]
            _closing_price_usd = self.to_usd(equity_name, _closing_price, er)
            for _, amount in self.account_long[equity_name].items():
                aum += _closing_price_usd*amount

            for p, amount in self.account_short[equity_name].items():
                aum += (2*p-_closing_price_usd)*amount

        aum += self.cash
        self.aum = aum
        self.aum_plot[str(date)] = self.aum


    def decision_algo(self, equity_name, up, confidence, vwap, closing_price, rsi):
        _long = False
        _short = False
        #confidence_thres = {'8035 JT':0.62, '6920 JT':0.67, '6146 JT':0.64, '7735 JT':0.63, '6857 JT':0.59, '240810 KS':0.8, '084370 KS':0.65, '688012 CH':0.81, 'LRCX US':0.56, 'AMAT US': 0.59, 'TER US':0.58, 'ASML NA':0.55}
        #if confidence > confidence_thres[equity_name]:
        if confidence >= 0.5:
            # long
            if closing_price < vwap and up and rsi > self.args.rsi_long_thres:
            #if closing_price < vwap and up:
            #if rsi > self.args.rsi_long_thres:
                _long = True
            elif closing_price > vwap and not up and rsi < self.args.rsi_short_thres:
            #elif closing_price > vwap and not up:
            #elif rsi < self.args.rsi_short_thres:
                _short = True
            else:
                pass

        return _long, _short

    @staticmethod
    def to_usd(equity_name, price, er):
        country = equity_name[-2:]
        if country == 'US':
            er = 1
        elif country == 'KS':
            er = er[0]
        elif country == 'JT':
            er = er[1]
        elif country == 'CH':
            er = er[2]
        elif country == 'NA':
            er = er[3]
        else:
            raise NotImplementedError
        return price/er



        
class Predictor:
    def __init__(self, args, model_pth):
        self.args = args
        self.load_model(model_pth) 
        self.softmax = nn.Softmax(dim=1)
        
    def load_model(self, model_pth):
        self.models = []
        dicts = torch.load(model_pth)
        print(f'loading model dictionary from {model_pth}')
        m_dicts = dicts['model']

        for _dict in m_dicts:
            model, _, _ = get_model(self.args)
            model.load_state_dict(_dict)
            model = model.eval()
            self.models.append(model)

    def predict(self, x):
        preds = []
        for model in self.models:
            pred = model(x)
            pred = pred.view(-1,2)
            preds.append(pred)
        pred = torch.stack(preds).mean(0)
        pred = self.softmax(pred)
        return pred


if __name__=='__main__':
    args_dir = './config_classifier_ampm.yaml'
    with open(args_dir, 'r') as fp:
        args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

    args.ntarget = [4]
    if 'end_date' not in args:
        args.end_date = None
    model_pth = './output/ampm_ntarget4_1year/best_model.pth'

    #long_thres_list = [55, 60, 65, 70]
    #short_thres_list = [40, 45, 50]
    long_thres_list = [60]
    short_thres_list = [40]

    max_aum = 0
    best_long_thres = 0
    best_short_thres = 0

    for long_thres in long_thres_list:
        for short_thres in short_thres_list:
            args.rsi_long_thres = long_thres
            args.rsi_short_thres = short_thres

            tester = BackTester(args, model_pth, 1000000, 0.0005)
            aum = tester.run(end_date=args.end_date, do_analysis=False)
            if max_aum < aum:
                max_aum = aum
                best_long_thres = long_thres
                best_short_thres = short_thres


    # best rsi threshold
    args.rsi_long_thres = best_long_thres
    args.rsi_short_thres = best_short_thres
    tester = BackTester(args, model_pth, 1000000, 0.0005)
    aum = tester.run(end_date=args.end_date, do_analysis=True)
    print(f'best rsi long thres: {best_long_thres}')
    print(f'best rsi short thres: {best_short_thres}')


        


