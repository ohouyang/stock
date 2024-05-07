import pandas as pd
import numpy as np
# from helper import *
import gzip
import _pickle as cPickle
import itertools
    
class FactorData(pd.DataFrame): # 这个类接收了一个dataframe
    
    @property
    def _constructor(self):
        return FactorData

    @property
    def _constructor_sliced(self):
        return pd.Series
    
    @property
    def fdate(self):
        return self._fdate

    @fdate.setter
    def fdate(self, value):
        self._fdate = value
    
    @property
    def fproduct(self):
        return self._fproduct

    @fproduct.setter
    def fproduct(self, value):
        self._fproduct = value
    
    @property
    def fHEAD_PATH(self):
        return self._fHEAD_PATH

    @fHEAD_PATH.setter
    def fHEAD_PATH(self, value):
        self._fHEAD_PATH = value
    
    def __getitem__(self, key):
        try:
            s = super().__getitem__(key)
        except KeyError:
            s = load(self._fHEAD_PATH+"/tmp pkl/"+self._fproduct+"/"+key+"/"+self._fdate)
            self[key] = s
        return s
        
        
import inspect
from collections import OrderedDict

class factor_template(object):
    factor_name = ""
    
    params = OrderedDict([
        ("period", np.power(2, range(10,13)))
    ])
    
    def formula(self):
        pass  
    
    def form_info(self):
        return inspect.getsource(self.formula)
    
    def info(self):
        info = ""
        info = info + "factor_name:\n"
        info = info +self.factor_name+"\n"
        info = info +"\n"
        info = info + "formula:\n"
        info = info +self.form_info()+"\n"
        info = info +"\n"
        info = info + "params:\n"
        for key in self.params.keys():
            info = info+"$"+key+":"+str(self.params.get(key))+"\n"
        return info
        
    def __repr__(self):
        return self.info()
    
    def __str__(self):
        return self.info()

def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)
    

def build_simple_signal(file_name, signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    
    data = load(file_name)
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        
        path = HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name[-12:]
        S = signal_list.formula(data, *cartesian)
        save(S, path)
        

def build_range_signal(file_name, signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    
    data = load(file_name)
    for cartesian in itertools.product(*signal_list.params.values()): #资料来源：https://blog.csdn.net/qq_36387683/article/details/109033312
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        #signal_names.append(signal_name)
        path = HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name[-12:]
        S = signal_list.formula(data, *cartesian)
        save(S, path)
        

def build_composite_signal(file_name, signal_list, product, HEAD_PATH, n=12): # n的默认值为12
    keys = list(signal_list.params.keys()) # keys中的值为 ['period']
    raw_data = load(file_name) # raw_data是一个dataframe
    data = FactorData(raw_data) # 这里用到了一个类
    data.fdate = file_name[-n:] # 在这里n的值为8 截取的是file_name字符串的后八位
    data.fproduct = product
    data.fHEAD_PATH = HEAD_PATH # 为类的两个属性赋值
    for cartesian in itertools.product(*signal_list.params.values()): # 该循环经历了三次 (1024,) (2048,) (4096,) cartesian取这三个值
        signal_name = signal_list.factor_name # 传入参数是一个类，该类的factor_name属性为  nr.period
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))  #这个循环的作用是把 nr.period字符串更改为 nr.1024 nr.2048 nr.4096
        path = HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name[-n:]
        S = signal_list.formula(data, *cartesian) 
        save(S, path)

def construct_composite_signal(dire_signal, range_signal, period_list, good_night_list, CORE_NUM, 
                                       product, HEAD_PATH, min_pnl=2,period=4096, tranct=0.25e-4, tranct_ratio=True):
    from collections import OrderedDict
    class foctor_xx_period(factor_template):
        factor_name = dire_signal+"."+range_signal+".period"
        params = OrderedDict([
            ("period", period_list)
        ])
        def formula(self, data, period):
            return (data[dire_signal+"."+str(period)]*data[range_signal+"."+str(period)]).values
    xx = foctor_xx_period()
    create_signal_path(xx, product, HEAD_PATH)
    parLapply(CORE_NUM, good_night_list, build_composite_signal, 
          signal_list=xx, product=product, HEAD_PATH=HEAD_PATH);
    new_signal = dire_signal+"."+range_signal+"."+str(period)
    good_night_files = np.array([x[-12:] for x in good_night_list])
    all_signal = get_all_signal(good_night_files, product, new_signal, period)
    open_thre = np.quantile(abs(all_signal), np.append(np.arange(0.991, 0.999, 0.001),
                                                  np.arange(0.9991,0.9999,0.0001)))
    thre_mat = pd.DataFrame(data=OrderedDict([("open", open_thre), ("close", -open_thre)]))
    print("reverse=1")
    signal_stat = get_signal_stat(new_signal, thre_mat, product, good_night_files,
                               reverse=1, tranct=tranct, tranct_ratio=tranct_ratio, min_pnl = min_pnl, CORE_NUM=CORE_NUM);
    print("reverse=-1")
    signal_stat = get_signal_stat(new_signal, thre_mat, product, good_night_files,
                               reverse=-1, tranct=tranct, tranct_ratio=tranct_ratio, min_pnl = min_pnl, CORE_NUM=CORE_NUM);
    
