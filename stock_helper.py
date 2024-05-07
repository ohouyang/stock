from product_info import *
from scipy.stats import kurtosis
from scipy.stats import skew
## path of our program
#HEAD_PATH = "/mnt/hgfs/intern"
HEAD_PATH = "F:/BigData/data"
SAVE_PATH = "F:/BigData/data"



## path of data
DATA_PATH = HEAD_PATH + "/stock pkl/"


## get all the dates of product
import os
def get_dates(product):
    return list(map(lambda x: x[:8] ,os.listdir(DATA_PATH + product)))

## get the data of product of specific date
import pandas as pd
import _pickle as cPickle
import gzip
#import lz4.frame
def get_data(product, date):
    data = load(DATA_PATH + product+"/"+date+".pkl")
    return data


def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)

        
import dask
from dask import compute, delayed
def parLapply(CORE_NUM, iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result
        
        
## returns 0 if the numerator or denominator is 0
import numpy as np
import warnings
def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x,y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        if hasattr(x, "__len__"):
            res = np.zeros(len(x))
        else:
            res = 0
        
    return res

## function to calculate t-stat
def get_t_value(train_mat, signal, response):
    beta = np.sum(train_mat[signal]*train_mat[response])/sum(train_mat[signal]**2) ## regressio coef
    sigma = np.sqrt(np.sum((train_mat[signal]*beta-train_mat[response])**2) / (len(train_mat)-1)) 
    v = np.sqrt(np.sum(train_mat[signal]**2)) ## sigma/v is the standard devication of beta_hat
    return beta/sigma*v

import statsmodels.formula.api as smf

## forward selection of signals
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    chosen_signals = []
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} - 1".format("data['"+response+"']",
                                           ' + '.join(selected + ["data['"+candidate+"']"]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append("data['"+best_candidate+"']")
            current_score = best_new_score
            chosen_signals.append(best_candidate)
    formula = "{} ~ {} - 1".format("data['"+response+"']",
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return OrderedDict([("model", model), ("chosen.signals", chosen_signals)])

def moving_average(a, n=3) :
    ret_sum = np.cumsum(a, dtype=float)
    ret = a
    ret[n:] = (ret_sum[n:] - ret_sum[:-n])/n
    return ret

def ewma(x, halflife, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        lamb=1 - 0.5**(1 / halflife)
        aa=1-np.power(1-lamb, xx)*(1-lamb)
        bb=s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]

def ewma_lambda(x, lambda_, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        aa=1-np.power(1-lambda_, xx)*(1-lambda_)
        bb=s.ewm(alpha=lambda_, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(alpha=lambda_, adjust=False).mean()[1:]

## moving sum of x
## we don't use rollSum because rollSum would make the first n data to be zero
def cum(x, n):
    sum_x = x.cumsum()
    sum_x_shift = sum_x.shift(n)
    sum_x_shift[:n]= 0
    return sum_x - sum_x_shift


def sharpe(x):
    return zero_divide(np.mean(x)* np.sqrt(250), np.std(x, ddof=1))

def drawdown(x):
    y = np.cumsum(x)
    return np.max(y)-np.max(y[-1:])

def max_drawdown(x):
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y)-y)

from collections import OrderedDict
def get_hft_summary(result, thre_mat):
    n_thre = np.shape(thre_mat)[0]
    all_pnl = np.zeros((0,n_thre))
    all_dates = np.array([])
    for i in range(len(result)):
        all_pnl =  np.concatenate((all_pnl,result[i]["all.pnl"]),axis=0)
        all_dates = np.append(all_dates, result[i]["date"])
    stat = result[0]["result"].iloc[:,2:]
    for i in range(1,len(result)):
        stat = stat+result[i]["result"].iloc[:,2:]
    stat["avg.ret"]=stat["ret"]/stat["num"]
    
    total_ret = all_pnl.sum(0)
    total_sharpe = np.zeros(n_thre)
    total_drawdown = np.zeros(n_thre)
    total_max_drawdown = np.zeros(n_thre)
    for i in range(n_thre):
        total_sharpe[i] = sharpe(all_pnl[:,i])
        total_drawdown[i] = drawdown(all_pnl[:,i])
        total_max_drawdown[i] = max_drawdown(all_pnl[:,i])
    final_result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"]), ("close", thre_mat["close"]), ("num", stat["num"]),
                                                  ("avg.ret", stat["avg.ret"]), ("total.ret",total_ret), ("sharpe", total_sharpe),
                                                  ("drawdown", total_drawdown), ("max.drawdown", total_max_drawdown),
                                                 ("mar", total_ret/total_max_drawdown)]), 
                                index=thre_mat.index)
    return OrderedDict([("final.result", final_result), ("daily.pnl", all_pnl), ("date", all_dates)])

def get_sample_signal(good_night_files, sample, product, signal_list, period, daily_num):
    n_samples = sum(daily_num[sample])
    n_days = sum(sample)
    n_signal = len(signal_list)
    all_signal =  np.ndarray(shape=(int(n_samples),n_signal))
    cur = 0
    for file in good_night_files[sample]:
        data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
        chosen = (np.arange(sum(data["good"]))+1) % period==0
        n_chosen = sum(chosen)
        for i in range(n_signal):
            signal_name = signal_list[i]
            S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
            S = S[data["good"]]
            signal = S[(np.arange(len(S))+1) % period == 0]
            signal[np.isnan(signal)] = 0 ## the ret.cor has some bad records
            signal[np.isinf(signal)] = 0 ## the ret.cor has some bad records
            all_signal[cur:(cur+n_chosen),i] = signal
        cur = cur+n_chosen
    all_signal = pd.DataFrame(all_signal, columns=signal_list)
    return all_signal


from collections import OrderedDict
def get_signal_pnl(file, product, signal_name, thre_mat, reverse=1,  buy_tranct=1.5e-4, sell_tranct=11.5e-4,
                   max_spread=0.011,
                   HEAD_PATH="F:/BigData/data", SAVE_PATH="F:/BigData/data", atr_filter=0):
    ## load data
    data = load(HEAD_PATH+"/stock pkl/"+product+"/"+file)
    S = load(SAVE_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    pred = S*reverse
    atr = load(SAVE_PATH+"/tmp pkl/"+product+"/"+"atr.4096"+"/"+file)
    #n_bar = len(data)
    
    ## load signal
    
    ## we don't know the signal is positive correlated or negative correlated  
    #n_thre = len(thre_mat)
    date = np.array([x[0:10] for x in data["date.time"]])
    next_date = np.append(date[1:],'1')
    end_day = date!=next_date
    count = 0;
    n_day = sum(end_day)
    n_thre = np.shape(thre_mat)[0]
    all_pnl = np.zeros((n_day, n_thre))
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.ret", 0), ("ret", 0)]), 
                          index=thre_mat.index)

    cur_spread = data["ask1"]-data["bid1"]
    for thre in thre_mat.iterrows():
        count = count+1
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[atr<atr_filter]=0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0) & (cur_spread<max_spread)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0) & (cur_spread<max_spread)] = 0
        position_pos.ffill(inplace=True)
        pre_pos = position_pos.shift(1)
        notional_position_pos = pd.Series(data=0, index=data.index)
        notional_position_pos[position_pos==1] = 1
        notional_position_pos[(position_pos==1) & (pre_pos==1)] = np.nan
        notional_position_pos[(notional_position_pos==1)] = 1/data["next.ask"][(notional_position_pos==1)]
        notional_position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0) & (cur_spread<max_spread)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0) & (cur_spread<max_spread)] = 0
        position_neg.ffill(inplace=True)
        pre_neg = position_neg.shift(1)
        notional_position_neg = pd.Series(data=0, index=data.index)
        notional_position_neg[position_neg==-1] = -1
        notional_position_neg[(position_neg==-1) & (pre_neg==-1)] = np.nan
        notional_position_neg[(notional_position_neg==-1)] = -1/data["next.bid"][(notional_position_neg==-1)]
        notional_position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        notional_position = notional_position_pos+notional_position_neg
        #position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        notional_position.iloc[0] = 0
        notional_position.iloc[-2:] = 0
        #change_pos = position - position.shift(1)
        #notional_change_pos = notional_position-notional_position.shift(1)
        change_pos = notional_position-notional_position.shift(1)
        change_pos.iloc[0] = 0
        #notional_change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos>0
        change_sell = change_pos<0
        change_base[change_buy] = data["next.ask"][change_buy]*(1+buy_tranct)*data["adjust"]
        change_base[change_sell] = data["next.bid"][change_sell]*(1-sell_tranct)*data["adjust"]
        raw_pnl = -(change_base*change_pos).cumsum()+notional_position*data["wpr"]
        final_pnl = -sum(change_base*change_pos) ## total pnl, there is a negative sign, because selling get money and buying pay money
        turnover = sum(change_base*abs(change_pos))
        num = sum((position!=0) & (change_pos!=0)) ## number of trades
        hld_period = sum(position!=0)   ## holding period
        daily_pnl = raw_pnl[end_day].reset_index(drop=True)
        pnl = np.append(daily_pnl[0], np.diff(daily_pnl))
        all_pnl[:,thre[0]] = pnl
        if (num==0):
            result.loc[thre[0], ("num","avg.ret","ret")] = (0,0,0)
        else:
            result.loc[thre[0],("num", "avg.ret", "ret", )] = (num, final_pnl/num, final_pnl)
    return OrderedDict([("all.pnl", all_pnl), ("result", result), ("date", date[end_day])])

# def vanish_thre(x, thre):
#     x[np.abs(x)>thre] = 0
#     return x
def vanish_thre(x, thre):
    x[x>thre] = thre
    x[x<-thre] = -thre
    return x


def construct_composite_signal(dire_signal, range_signal, period_list, product_list, HEAD_PATH):
    from collections import OrderedDict
    class foctor_xx_period(factor_template):
        factor_name = dire_signal+"."+range_signal+".period" # 新因子名称
        params = OrderedDict([
            ("period", period_list)
        ]) 
        def formula(self, data, period):
            return (data[dire_signal+"."+str(period)]*data[range_signal+"."+str(period)]).values # 因子计算
    xx = foctor_xx_period()
    for product in product_list:
        create_signal_path(xx, product, HEAD_PATH)
        file_list = list(map(lambda x: DATA_PATH+product+"/"+x, os.listdir(DATA_PATH + product)))
        parLapply(CORE_NUM, file_list, build_composite_signal,signal_list=xx, product=product, HEAD_PATH=HEAD_PATH, n=8)

import itertools
def create_signal_path(signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        
        os.makedirs(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name, exist_ok=True)
        print(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name)

def get_signal_mat(signal_mat, signal_name, product, file_name, HEAD_PATH="F:/BigData/data"):
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name)
    S[np.isnan(S)] = 0
    if signal_mat is None:
        return S
    else:
        return np.vstack((signal_mat, S))

# def get_daily_pred(file_name, product, signal_list, coef, strat, HEAD_PATH):
#     signal_mat = functools.reduce(functools.partial(get_signal_mat, product=product, file_name=file_name, HEAD_PATH=HEAD_PATH), signal_list, None)
#     if len(coef) > 1:
#         S = np.dot(signal_mat.T, coef)
#     else:
#         S = signal_mat * coef
#     save(S, HEAD_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)      
    
def get_sample_signal(good_night_files, sample, product, signal_list, period, daily_num, HEAD_PATH):
    n_samples = sum(daily_num[sample])
    n_days = sum(sample)
    n_signal = len(signal_list)
    all_signal =  np.ndarray(shape=(int(n_samples),n_signal))
    cur = 0
    for file in good_night_files[sample]:
        good = load(HEAD_PATH+"/good pkl/"+product+"/"+file)
        chosen = (np.arange(sum(good))+1) % period==0
        n_chosen = sum(chosen)
        for i in range(n_signal):
            signal_name = signal_list[i]
            S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
            S = S[good]
            signal = S[(np.arange(len(S))+1) % period == 0]
            signal[np.isnan(signal)] = 0 ## the ret.cor has some bad records
            signal[np.isinf(signal)] = 0 ## the ret.cor has some bad records
            all_signal[cur:(cur+n_chosen),i] = signal
        cur = cur+n_chosen
    all_signal = pd.DataFrame(all_signal, columns=signal_list)
    return all_signal

def get_range_pos(wpr, min_period, max_period, period):
    return ewma(zero_divide(wpr-min_period, max_period-min_period), period, adjust=True) - 0.5


import dask
from dask import compute, delayed
import matplotlib.pyplot as plt
import functools

from collections import OrderedDict
def get_signal_train_stat(signal_name, thre_mat, product, all_dates, CORE_NUM, split_str="2018", reverse=1, tranct=1.1e-4, 
                    max_spread=0.61, tranct_ratio=True, min_pnl=2, min_num=20, atr_filter=0):
    train_sample = all_dates<split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                 reverse=reverse, tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio, atr_filter=atr_filter)
        train_result = compute([delayed(f_par)(file) for file in all_dates[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    return train_stat

from collections import OrderedDict

# def get_signal_stat(signal_name, thre_mat, product, all_dates, CORE_NUM, split_str="2018", reverse=1, tranct=1.1e-4, 
#                     max_spread=0.61, tranct_ratio=True, min_pnl=2, min_num=20, atr_filter=0, HEAD_PATH="d:/intern", SIGNAL_PATH="d:/intern"):
#     train_sample = all_dates<split_str
#     test_sample = all_dates>split_str
#     with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
#         f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
#                                  reverse=reverse, tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio, 
#                                   HEAD_PATH=HEAD_PATH, SIGNAL_PATH=SIGNAL_PATH, atr_filter=atr_filter)
#         train_result = compute([delayed(f_par)(file) for file in all_dates[train_sample]])[0]
#     train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
#     with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
#         f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
#                                   reverse=reverse, tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio, 
#                                   HEAD_PATH=HEAD_PATH, SIGNAL_PATH=SIGNAL_PATH, atr_filter=atr_filter)
#         test_result = compute([delayed(f_par)(file) for file in all_dates[test_sample]])[0]
#     test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
#     return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat)])

def get_signal_stat(signal_name, thre_mat, product, all_dates, CORE_NUM, split_str="2018", reverse=1, tranct=1.1e-4, 
                    max_spread=0.61, tranct_ratio=True, min_pnl=2, min_num=20, atr_filter=0, HEAD_PATH="F:/BigData/data", SIGNAL_PATH="F:/BigData/data"):
    train_sample = all_dates<split_str
    test_sample = all_dates>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                 reverse=reverse, max_spread=max_spread,
                                  HEAD_PATH=HEAD_PATH, atr_filter=atr_filter)
        train_result = compute([delayed(f_par)(file) for file in all_dates[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat)
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                  reverse=reverse, max_spread=max_spread,
                                  HEAD_PATH=HEAD_PATH, atr_filter=atr_filter)
        test_result = compute([delayed(f_par)(file) for file in all_dates[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat)
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat)])

def rsi(ret, period):
    abs_move = np.abs(ret)
    up_move = np.maximum(ret, 0)
    up_total = ewma(up_move, period, adjust=True)
    move_total = ewma(abs_move, period, adjust=True)
    rsi = zero_divide(up_total, move_total) - 0.5
    return rsi

from collections import OrderedDict
def get_list_signal_stat(signal_name, thre_mat_list, product_list, atr_filter_list,
                    split_str="2018", min_pnl=2, min_num=20, reverse=1):
    CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
    train_trade_stat = dict([]) ## training result statistics
    print("training")
    for product in product_list:
        print(product)
        all_dates = np.array(list(map(lambda x: x,os.listdir(DATA_PATH + product))))
        train_sample = all_dates<split_str # 获得训练样本
        test_sample = all_dates>split_str # 获得测试样本
        print(sum(train_sample), sum(test_sample))
        thre_mat = thre_mat_list[product]
        atr_filter = atr_filter_list[product]
        with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
            f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                 reverse=1,
                                HEAD_PATH=HEAD_PATH, SAVE_PATH=SAVE_PATH,atr_filter=atr_filter)
            train_result = compute([delayed(f_par)(file) for file in np.array(all_dates)[train_sample]])[0]
        trade_stat = get_hft_summary(train_result, thre_mat)
        train_trade_stat[product] = trade_stat
    print("testing")
    test_trade_stat = dict([])
    for product in product_list:
        all_dates = np.array(list(map(lambda x: x,os.listdir(DATA_PATH + product))))
        print(product)
        train_sample = np.array(all_dates)<split_str ## training samples
        test_sample = np.array(all_dates)>split_str ## test samples
        thre_mat = thre_mat_list[product]
        reverse = reverse_list[product]
        atr_filter = atr_filter_list[product]
        with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
            f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                 reverse=1,
                                HEAD_PATH=HEAD_PATH, SAVE_PATH=SAVE_PATH,atr_filter=atr_filter)
            result = compute([delayed(f_par)(file) for file in np.array(all_dates)[test_sample]])[0]
        trade_stat = get_hft_summary(result, thre_mat)
        test_trade_stat[product] = trade_stat
    result=dict([])
    result["train_trade_stat"] = train_trade_stat # 保存训练相关数据
    result["test_trade_stat"] = test_trade_stat # 保存测试相关数据
    if reverse==-1: # 根据reverse的取值把运行结果保存到不同的路径下面
        save(result, HEAD_PATH+"/"+signal_name+".result.pkl") # 反转策略
    else:
        save(result, HEAD_PATH+"/"+signal_name+".pos.result.pkl") # 趋势策略

        
# def get_list_signal_stat(signal_name, thre_mat, product_list, all_dates, split_str="2018", reverse=1,
#                     min_pnl=2, min_num=20, atr_filter=20):
#     CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
#     train_sample = np.array(all_dates)<split_str
#     test_sample = np.array(all_dates)>split_str
#     date_str = [n[0:8] for n in all_dates]
#     format_dates = np.array([pd.to_datetime(d) for d in date_str])
#     train_trade_stat = dict([])
#     print("training")
#     for product in product_list:
#         spread = product_info[product]["spread"]
#         tranct = product_info[product]["tranct"]
#         tranct_ratio = product_info[product]["tranct.ratio"]
#         with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
#             f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
#                                  reverse=reverse, tranct=tranct, max_spread=spread*1.1, tranct_ratio=tranct_ratio, atr_filter=atr_filter)
#             train_result = compute([delayed(f_par)(file) for file in np.array(all_dates)[train_sample]])[0]
#         trade_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
#         train_trade_stat[product] = trade_stat
#     print("testing")
#     test_trade_stat = dict([])
#     for product in product_list:
#         spread = product_info[product]["spread"]
#         tranct = product_info[product]["tranct"]
#         tranct_ratio = product_info[product]["tranct.ratio"]
#         with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
#             f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
#                                      reverse=reverse, tranct=tranct, max_spread=spread*1.1, tranct_ratio=tranct_ratio, atr_filter=atr_filter)
#             result = compute([delayed(f_par)(file) for file in np.array(all_dates)[test_sample]])[0]
#         trade_stat = get_hft_summary(result, thre_mat, sum(test_sample))
#         test_trade_stat[product] = trade_stat
#     result=dict([])
#     result["train_trade_stat"] = train_trade_stat
#     result["test_trade_stat"] = test_trade_stat
#     if reverse==-1:
#         save(result, HEAD_PATH+"/"+signal_name+".result.pkl")
#     else:
#         save(result, HEAD_PATH+"/"+signal_name+".pos.result.pkl")
        

## get train and test result, choose good strategies, plot the portfolio and result of each product

def get_list_signal_result(signal_name, product_list, split_str="2018", reverse=1,
                           min_pnl=2, min_num=20, atr_filter=20):
    if reverse==-1:
        result = load(HEAD_PATH+"/"+signal_name+".result.pkl") ## load reversal strat statistics
    else:
        result = load(HEAD_PATH+"/"+signal_name+".pos.result.pkl") ## load trend strat statistics
    train_trade_stat = result["train_trade_stat"]
    test_trade_stat = result["test_trade_stat"]
    train_sample = np.array(all_dates)<split_str
    test_sample = np.array(all_dates)>split_str
    i = 0
    for product in product_list:
        trade_stat = train_trade_stat[product]
        ## two criterion: minimum averge pnl (unit is spread), monimum number of trades
        good_strat = (trade_stat["final.result"]["avg.ret"]>min_pnl) & (trade_stat["final.result"]["num"]>min_num)
        if sum(good_strat)>0:
            train_pnl = trade_stat["daily.pnl"][:, good_strat].sum(axis=1)/sum(good_strat)
            trade_stat = test_trade_stat[product]
            test_pnl = trade_stat["daily.pnl"][:, good_strat].sum(axis=1)/sum(good_strat)
            print(product, "train sharpe ", sharpe(train_pnl), "test sharpe ", sharpe(test_pnl))
            i = i+1        
        
def get_good(date, product, HEAD_PATH, SAVE_PATH):
    data = load(HEAD_PATH+"/pkl tick/"+product+"/"+date)
    good = data["good"]
    save(good, SAVE_PATH+"/good pkl/"+product+"/"+date)        
        
def get_all_signal(file_list, product, signal_name, period, HEAD_PATH="F:/BigData/data", SIGNAL_PATH="F:/BigData/data"):
    n_files = len(file_list)
    all_signal = np.array([])
    for file in file_list:
        S = load(SIGNAL_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
        signal = S
        chosen = (np.arange(len(signal))+1) % period==0
        all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
    return all_signal

## parallel generate the distribution of a signal
# def par_get_all_signal(signal_name, file_list, product, period, SAVE_PATH="F:/BigData/data"):
#     n_files = len(file_list)
#     all_signal = np.array([])
#     for file in file_list:
#         S = load(SAVE_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file) ## signal
#         good = load(SAVE_PATH+"/good pkl/"+product+"/"+file) ## good singal
#         signal = S[good]
#         moving_average(signal,period)
#         chosen = (np.arange(len(signal))+1) % period==0
#         all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
#     save(all_signal, SAVE_PATH+"/all signal/"+product+"."+signal_name+".pkl")
def par_get_all_signal(signal_name, file_list, product, period, HEAD_PATH="F:/BigData/data", SAVE_PATH="F:/BigData/data"):
    n_files = len(file_list)
    all_signal = np.array([])
    for file in file_list:
        S = load(SAVE_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file) ## signal
        signal = S
        chosen = (np.arange(len(signal))+1) % period==0
        all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
    save(all_signal, SAVE_PATH+"/all signal/"+product+"."+signal_name+".pkl")

def par_get_arb_all_signal(signal_name, file_list, product_x, product_y, period, HEAD_PATH="F:/BigData/data"):
    n_files = len(file_list)
    all_signal = np.array([])
    for file in file_list:
        S_x = load(HEAD_PATH+"/tmp pkl/"+product_x+"/"+signal_name+"/"+file)
        S_y = load(HEAD_PATH+"/tmp pkl/"+product_y+"/"+signal_name+"/"+file)
        [time_x, time_y] = load(HEAD_PATH+"/comb time/"+product_x+"_"+product_y+"/"+file)
        signal = S_x[time_x] - S_y[time_y]
        chosen = (np.arange(len(signal))+1) % period==0
        all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
    save(all_signal, HEAD_PATH+"/all signal/"+product_x+"_"+product_y+"."+signal_name+".pkl")
    
def add_min_max(file, period_list):
    data = load(file)
    for period in period_list:
        data["min."+str(period)] = data["wpr"].rolling(period).min()
        data.iloc[:perio1, ("min."+str(period))] = np.minimum.accumulate(data["wpr"].iloc[:period])
        #data.iloc[:period-1, ("min."+str(period))] = data["wpr"][0]
        
        data["max."+str(period)] = data["wpr"].rolling(period).max()
        data.iloc[:period, ("max."+str(period))] = np.maximum.accumulate(data["wpr"].iloc[:period])
        #data.iloc[:period-1, ("max."+str(period))] = data["wpr"][0]
    
    save(data, file)
    #print("end ", file)
    
    
def get_signal_pnl_close(file, product, signal_name, thre_mat, reverse=1, rebate=0, HEAD_PATH="F:/BigData/data"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    ## load signal
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    ## we don't know the signal is positive correlated or negative correlated  
    pred = S*reverse
    pred = pred[data["good"]]
    data = data[data["good"]]
    ## load product info
    tranct = product_info[product]["tranct"]*(1-rebate)
    min_spread = product_info[product]["spread"]+0.001
    close = product_info[product]["close"]*(1-rebate)
    tranct_ratio = product_info[product]["tranct.ratio"]
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    bid_ask_spread = data["ask"]-data["bid"]
    next_spread = bid_ask_spread.shift(-1)
    next_spread.iloc[-1] = bid_ask_spread.iloc[-1]
    not_trade = (data["time"]=="10:15:00") | (data["time"]=="11:30:00") | (data["time"]=="15:00:00") | (bid_ask_spread>min_spread) | (next_spread>min_spread)

    for thre in thre_mat.iterrows():
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[not_trade] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        #change_base = pd.Series(data=0, index=data.index)
        
        pre_pos = position.shift(1)
        pre_pos.iloc[0] = 0
        open_buy = (pre_pos<=0) & (position>0)
        open_sell = (pre_pos>=0) & (position<0)
        close_buy = (pre_pos<0) & (position>=0)
        close_sell = (pre_pos>0) & (position<=0)
        open_buy_pnl = pd.Series(data=0, index=data.index)
        open_sell_pnl = pd.Series(data=0, index=data.index)
        close_buy_pnl = pd.Series(data=0, index=data.index)
        close_sell_pnl = pd.Series(data=0, index=data.index)
        
        if tranct_ratio:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy]*(1+tranct)
            open_sell_pnl[open_sell] = data["next.bid"][open_sell]*(1-tranct)
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy]*(1+close)
            close_sell_pnl[close_sell] = data["next.bid"][close_sell]*(1-close)
        else:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy]-tranct
            open_sell_pnl[open_sell] = data["next.bid"][open_sell]-tranct
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy]-close
            close_sell_pnl[close_sell] = data["next.bid"][close_sell]-close
        final_pnl = sum(open_buy_pnl+open_sell_pnl+close_buy_pnl+close_sell_pnl)
        num = sum((position!=0) & (change_pos!=0))
        
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            
    return result

def fast_roll_var(x, period):
    x_ma = cum(x,period)/period
    x2 = x*x
    x2_ma = cum(x2,period)/period
    var_x = x2_ma-x_ma*x_ma
    var_x[var_x<0] = 0
    return(var_x)

def fast_roll_cor_ewma(x, y, period):
    x_ma = ewma(x,period)
    x2 = x*x
    x2_ma = ewma(x2,period)
    var_x = x2_ma-x_ma*x_ma
    var_x[var_x<0] = 0
    y_ma = ewma(y,period)
    y2 = y*y
    y2_ma = ewma(y2,period)
    var_y = y2_ma-y_ma*y_ma
    var_y[var_y<0] = 0
    upper = ewma(x*y, period) - x_ma*y_ma
    result = zero_divide(upper, np.sqrt(var_x*var_y))
    return (result)


def fast_roll_cor(x, y, period):
    x_ma = cum(x,period)/period
    x2 = x*x
    x2_ma = cum(x2,period)/period
    var_x = x2_ma-x_ma*x_ma
    var_x[var_x<0] = 0
    y_ma = cum(y,period)/period
    y2 = y*y
    y2_ma = cum(y2,period)/period
    var_y = y2_ma-y_ma*y_ma
    var_y[var_y<0] = 0
    #upper = (x-x_ma)*(y-y_ma)
    #result = zero_divide(cum(upper, period), np.sqrt(var_x*var_y))/period
    upper = cum(x*y, period) - period*x_ma*y_ma
    result = zero_divide(upper, np.sqrt(var_x*var_y))/period
    return (result)

def fast_ret_cor(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    rolling_cor = fast_roll_cor(pre_ret, ret, period)
    rolling_cor.iloc[:period-1] = 0
    return rolling_cor

def fast_ret_cor_ewma2(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period)*ewma(ret,period)*period
    result.iloc[:period-1]=0
    return result

def fast_ret_cor_ewma(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period)*ewma(ret,period)*period
    result = np.asarray(result)
    return result

def vol_cor(ret, qty, period):
    result = fast_roll_cor_ewma(qty, ret, period)*ewma(np.abs(ret),period)*period
    return result

def check_strat_prob(train_pnl, test_pnl, num=10000):
    random.seed([100])
    aa = np.random.standard_normal(num).reshape(-1,num)
    aa.sum(axis=1)
    
def fcum(x, n, fill=0):
    return pd.Series(data=cum(pd.concat((x, pd.Series(np.repeat(fill, n))), ignore_index=True), n).shift(-n)[:-n].values, index=x.index)

   
# def get_daily_pred(file_name, product, signal_list, coef, strat, HEAD_PATH):
#     signal_mat = load(HEAD_PATH+"/signal mat pkl/"+product+"/"+file_name)
#     if len(coef)>1:
#         S = np.dot(signal_mat.T, coef)
#     else:
#         S = signal_mat * coef
#     save(S, HEAD_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)

from scipy.optimize import minimize
def TotalTRC(x, Cov):
    x = np.append(x, 1-np.sum(x))
    TRC = np.prod((np.dot(Cov, x), x), axis=0)
    if np.sum(x<0)>0: 
        return 10**12
    else:
        return np.sum((TRC[:, None] - TRC) ** 2)
    
def risk_parity(Sub, only_diag=False, maxiter=9999):
    m = Sub.shape[1]
    Cov = np.cov(Sub, rowvar=False)
    if only_diag:
        Cov = np.diag(np.diag(Cov))
    res = minimize(functools.partial(TotalTRC, Cov=Cov), np.repeat(1/m, m-1), method="Nelder-Mead", options={'xtol': 1e-6, "maxiter": maxiter, "disp":True})
    w = np.append(res["x"], 1-np.sum(res["x"]))
    #res = nelder_mead(functools.partial(TotalTRC, Cov=Cov), np.repeat(1/m, m-1), step=1e-3, no_improve_thr=1e-05)
    #w = np.append(res[0], 1-np.sum(res[0]))
    return w

def get_signal_stat_close(signal_name, thre_mat, product, good_night_files, split_str="2018", reverse=1, min_pnl=2, min_num=10, rebate=0, CORE_NUM=4):
    train_sample = good_night_files<split_str
    test_sample = good_night_files>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(ll_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse, rebate=rebate)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"]>=min_pnl) & (train_stat["final.result"]["num"]>=min_num)
    if sum(good_strat)==0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse, rebate=rebate)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])    


def get_signal_stat_roll(signal_name, thre_mat, product, good_night_files, train_sample, test_sample,
                         reverse=1, min_pnl=2, min_num=10, CORE_NUM=4):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, len(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"]>=min_pnl) & (train_stat["final.result"]["num"]>=min_num)
    if sum(good_strat)==0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, len(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])    



def evaluate_signal(signal, all_dates, product, min_pnl, min_num, 
                    CORE_NUM, HEAD_PATH="F:/BigData/data", SIGNAL_PATH="F:/BigData/data", period=4096, split_str="2018", tranct=1.1e-4, 
                    max_spread=0.61, tranct_ratio=True, atr_filter=0, save_path="signal result",reverse=0):
    signal_name = signal+"."+str(period)
    all_signal = load(SIGNAL_PATH+"/all signal/"+product+"."+signal_name+".pkl")
    tranct = 0.05 # 交易手续费，万分之五
    tranct_ratio = False # True:按合约收费  False:固定收费 
    open_list = np.quantile(abs(all_signal), np.append(np.arange(0.991,0.999,0.001),np.arange(0.9991,0.9999,0.0001)))
    thre_list = []
    for cartesian in itertools.product(open_list, np.array([0.2, 0.4, 0.6, 0.8, 1.0])):
        thre_list.append((cartesian[0], -cartesian[0] * cartesian[1]))
    thre_list = np.array(thre_list)
    thre_mat = pd.DataFrame(data=OrderedDict([("open", thre_list[:, 0]), ("close", thre_list[:, 1])]))
    if reverse>=0:
        print("reverse=1")
        trend_signal_stat = get_signal_stat(signal_name, thre_mat, product, all_dates, CORE_NUM, split_str=split_str, reverse=1, tranct=tranct, 
                                   max_spread=max_spread, tranct_ratio=tranct_ratio, min_pnl=min_pnl, min_num=min_num, atr_filter=atr_filter, HEAD_PATH=HEAD_PATH, SIGNAL_PATH=SIGNAL_PATH)
    if reverse<=0:
        print("reverse=-1")
        reverse_signal_stat = get_signal_stat(signal_name, thre_mat, product, all_dates, CORE_NUM, split_str=split_str, reverse=-1, tranct=tranct, 
                        max_spread=max_spread, tranct_ratio=tranct_ratio, min_pnl=min_pnl, min_num=min_num, atr_filter=atr_filter, HEAD_PATH=HEAD_PATH, SIGNAL_PATH=SIGNAL_PATH)
    if reverse==0:
        stat_result = OrderedDict([("trend.signal.stat", trend_signal_stat), ("reverse.signal.stat", reverse_signal_stat)])    
        save(stat_result, HEAD_PATH+"/"+save_path+"/"+product+"."+signal_name+".pkl")
    elif reverse==1:
        save(trend_signal_stat, HEAD_PATH+"/"+save_path+"/"+product+"."+signal_name+".trend.pkl")
    elif reverse==-1:
        save(reverse_signal_stat, HEAD_PATH+"/"+save_path+"/"+product+"."+signal_name+".reverse.pkl")

        
## get the signal performance
## including trend and reverse signals
def get_signal_performance_result(all_period_signal, signal_dire, period, product_list, train_sample, test_sample):
    trend_signal_result = pd.DataFrame(data=OrderedDict([("signal", all_period_signal), ("reverse",1),
                               ("num", 0), ("trainSharpe", 0), ("testSharpe", 0)]))
    reverse_signal_result = pd.DataFrame(data=OrderedDict([("signal", all_period_signal), ("reverse",-1),
                               ("num", 0), ("trainSharpe", 0), ("testSharpe", 0)]))
    n_signal = len(all_period_signal) ## number of all signals
    for k in range(n_signal):
        test_all_pnl = np.zeros([sum(test_sample), len(product_list)]) ## test pnl
        train_all_pnl = np.zeros([sum(train_sample), len(product_list)]) ## train pnl
        signal_name = all_period_signal[k]
        i = 0
        for product in product_list:
            stat_result = load(HEAD_PATH+"/" + signal_dire +"/"+product+"."+signal_name+".pkl") ## statistics of signal over a product
            trend_signal_stat = stat_result['trend.signal.stat'] ## trending statistics
            spread = product_info[product]["spread"]
            if tuple(trend_signal_stat.keys())[0]=='train.stat':
                train_stat = trend_signal_stat["train.stat"]
                test_stat = trend_signal_stat["test.stat"]
                #good_strat = trend_signal_stat["good.strat"]
                good_strat = (train_stat["final.result"]["avg.pnl"]>2*spread) & (train_stat["final.result"]["num"]>10) ## filter criterion
                if sum(good_strat)>2:
                    train_stat = trend_signal_stat["train.stat"]
                    test_stat = trend_signal_stat["test.stat"]
                    train_pnl = train_stat["daily.ret"].loc[:, good_strat].sum(axis=1)/sum(good_strat) ## get the daily return
                    #train_std = np.std(train_pnl)
                    #train_pnl = train_pnl/train_std
                    test_pnl = test_stat["daily.ret"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
                    #print(product, "train sharpe ", sharpe(train_pnl), "test sharpe ", sharpe(test_pnl))
                    test_all_pnl[:,i] = test_pnl
                    train_all_pnl[:,i] = train_pnl
                    i = i+1
            if i>0: ## if there are any good products
                train_portfolio = np.array(np.mean(train_all_pnl[:,0:i], axis=1)) ## training portfolio
                test_portfolio = np.array(np.mean(test_all_pnl[:,0:i], axis=1)) ## testing portfolio
                all_portfolio = np.append(train_portfolio, test_portfolio)
                trend_signal_result.loc[k, ("signal", "num", "trainSharpe", "testSharpe")] = (signal_name, i,  sharpe(train_portfolio),sharpe(test_portfolio))
        test_all_pnl = np.zeros([sum(test_sample), len(product_list)])
        train_all_pnl = np.zeros([sum(train_sample), len(product_list)])
        i = 0
        for product in product_list:
            stat_result = load(HEAD_PATH+"/"+signal_dire+"/"+product+"."+signal_name+".pkl")
            reverse_signal_stat = stat_result['reverse.signal.stat']
            spread = product_info[product]["spread"]
            if tuple(reverse_signal_stat.keys())[0]=='train.stat':
                #good_strat = reverse_signal_stat["good.strat"]
                train_stat = reverse_signal_stat["train.stat"]
                test_stat = reverse_signal_stat["test.stat"]
                good_strat = (train_stat["final.result"]["avg.pnl"]>2*spread) & (train_stat["final.result"]["num"]>10)
                if sum(good_strat)>2:
                    train_pnl = train_stat["daily.ret"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
                    #train_std = np.std(train_pnl)
                    #train_pnl = train_pnl/train_std
                    test_pnl = test_stat["daily.ret"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
                    test_all_pnl[:,i] = test_pnl
                    train_all_pnl[:,i] = train_pnl
                    i = i+1
            if i>0:
                train_portfolio = np.array(np.mean(train_all_pnl[:,0:i], axis=1))
                test_portfolio = np.array(np.mean(test_all_pnl[:,0:i], axis=1))
                all_portfolio = np.append(train_portfolio, test_portfolio)
                reverse_signal_result.loc[k, ("signal","num", "trainSharpe", "testSharpe")] = (signal_name, i, sharpe(train_portfolio),sharpe(test_portfolio))
    return OrderedDict([("trend.signal.stat", trend_signal_result), 
                        ("reverse.signal.stat", reverse_signal_result)])
                        
        
def par_get_daily_pred(file_name, product, coef, strat, HEAD_PATH, SAVE_PATH):
    signal_mat = load(HEAD_PATH+"/signal mat pkl/"+product+"/"+file_name)
    S = np.dot(signal_mat.transpose(),coef)
    save(S, SAVE_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)      

    
def get_daily_gbm(file_name, product, signal_list, model, strat, HEAD_PATH, SAVE_PATH, train_std, thre=float('Inf')):
    signal_mat = load(SAVE_PATH+"/signal mat pkl/"+product+"/"+file_name).T/train_std
    S = model.predict(signal_mat)
    S[np.abs(S)>thre] = 0
    save(S, SAVE_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)
    
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import lasso_path, LassoCV
from sklearn.preprocessing import StandardScaler
def get_multiple_lasso_roll_model(train_start, train_end, y_signal, n_signal, daily_ticks, all_mat, forward_len, strat, 
                          single_product, combine_product, period=4096, SAVE_PATH="F:/BigData/data"):
    product_list = np.append(single_product, combine_product)
    cum_daily_ticks = dict([])
    train_tick_start = dict([])
    train_tick_end = dict([])   
    test_tick_start = dict([])
    test_tick_end = dict([])   
    for product in product_list:
        cum_daily_ticks[product] = daily_ticks[product].cumsum()
        if train_start==0:
            train_tick_start[product] = 0
        else:
            train_tick_start[product] = int(cum_daily_ticks[product][train_start-1]+1)
        train_tick_end[product] = int(cum_daily_ticks[product][train_end]-1)
        test_tick_start[product] = int(train_tick_end[product]+2)
        test_tick_end[product] = int(cum_daily_ticks[product][train_end+1])
        
    coef_list = dict([])
    for product in single_product:
        x_train = all_mat[product].iloc[train_tick_start[product]:train_tick_end[product], :n_signal].values
        y_train = all_mat[product][y_signal][train_tick_start[product]:train_tick_end[product]]
        n_train = x_train.shape[0]
        scaler =  StandardScaler(copy=True, with_mean=False, with_std=True)
        scaler.fit(x_train)
        x_std = np.sqrt(scaler.var_)
        x_train_normal = scaler.transform(x_train)
        model = LassoCV(n_alphas=100, fit_intercept=False, cv=5, max_iter=10000).fit(x_train_normal, y_train)
        coef = model.coef_ / x_std
        coef_list[product] = coef
    train_std_mat = dict([])
    train_std_mat
    n_samples = x_train.shape[0]
    train_mat = np.zeros((0,n_signal))
    y_train = np.array([])
    for product in combine_product:
        x_train = all_mat[product].iloc[train_tick_start[product]:train_tick_end[product], :n_signal].values
        cur_y_train = all_mat[product][y_signal][train_tick_start[product]:train_tick_end[product]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        train_std_mat[product] = np.sqrt(scaler.var_)
        train_mat = np.append(train_mat,x_train, axis=0)
        y_train = np.append(y_train, cur_y_train)
    model = LassoCV(n_alphas=100, fit_intercept=False, cv=5, max_iter=10000).fit(train_mat, y_train)
    fit_coef = model.coef_
    for product in combine_product:
        coef_list[product] = fit_coef/train_std_mat[product]
    os.makedirs(SAVE_PATH+"/model", exist_ok=True)
    save(coef_list, SAVE_PATH+"/model/"+strat+".pkl")
    

    
def get_glmnet_ensemble_roll_model(train_start, train_end, forward_len, alpha=1, start_year=2018, period=2048):
    cum_daily_ticks = daily_ticks.cumsum()
    if train_start==0:
        train_tick_start = 0
    else:
        train_tick_start = cum_daily_ticks[train_start-1]+1
    train_tick_end = cum_daily_ticks[train_end]-1
    test_tick_start = train_tick_end+2
    test_tick_end = cum_daily_ticks[train_end+1]
    n_signal = len(signal_list)
    nfold = 10
    model_coef = np.zeros((n_signal, n_mod))
    for i_mod in range(n_mod):
        x_train = train_array[i_mod,:,:n_signal]
        y_train = train_array[i_mod,:,n_signal]
        n_train = x_train.shape[0]
        model = ElasticNetCV(l1_ratio=alpha, n_alphas=100, fit_intercept=False, cv=10, max_iter=1000).fit(x_train, y_train)
        model_coef[:,i_mod] = model.coef_
    coef = np.mean(model_coef, axis=1)
    if alpha==1:
        strat = "lasso.ensemble.roll."+str(start_year)+"."+str(period)
    elif alpha==0:
        strat = "ridge.ensemble.roll."+str(start_year)+"."+str(period)
    else:
        strat = "elastic.ensemble.roll."+str(start_year)+"."+str(period)
    os.makedirs(HEAD_PATH+"/roll model", exist_ok=True)
    os.makedirs(HEAD_PATH+"/roll model/"+product, exist_ok=True)
    save(model_coef, HEAD_PATH+"/roll model/"+product+"/"+strat+"."+str(train_start)+"."+str(train_end)+".pkl")
   