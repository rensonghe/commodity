#%%
import pandas as pd
import numpy as np
import os
import time
import datetime
# import xcsc_tushare as ts
import tushare as ts
#
pro = ts.pro_api('591e6891f9287935f45fc712bcf62335a81cd6829ce76c21c0fdf7b2')
# 上期所 SHF; 大商所 DCE; 郑商所 ZCE; 上海能源交易所 INE
future = 'SN'
trading_list = pro.fut_mapping(ts_code='SN.SHF')
year = 2022
exchange = 'shfe'
data_path = 'ctpdata'
#
trading_list['year'] = trading_list['trade_date'].str.extract('(.{0,4})')
trading_list['year'] = trading_list['year'].astype(int)
future_year = trading_list[trading_list['year']>=year]
future_year['contract'] = future_year['mapping_ts_code'].str.extract('(.{0,6})')
future_year['contract'] = future_year['contract'].str.lower()
#
date = []
g = os.walk('/home/SourceData/%s/%s/'%(exchange, data_path))

for path, dir_list, file_list in g:
    for dir_name in dir_list:
        date.append(dir_name)

        # print(dir_name)
#%%
all_data = pd.DataFrame()
all_data_1 = pd.DataFrame()
all_data_2 = pd.DataFrame()
for i in date:
    if i >='20220701' and i <='20221230':
        print(i)
        # for j in future_year['trade_date']:
        #     if i == j:
        #         contract = future_year[future_year['trade_date'] == i]['contract'].values[0]
        #         dir_file = '/home/SourceData/{}/{}/{}/{}_{}.csv'.format(exchange,data_path, i, i, contract)
        #         # print(dir_file)
        #         file_1 = pd.read_csv(dir_file)
        #         file_1 = file_1.iloc[:,1:]
        #         cols_1 = ['timestamp', 'price', 'high', 'low', 'averageprice', 'volume','amount', 'open_interest',
        #                 'bid_price1', 'bid_size1', 'ask_price1', 'ask_size1', 'bid_price2','bid_size2', 'ask_price2', 'ask_size2',
        #                 'bid_price3', 'bid_size3', 'ask_price3', 'ask_size3', 'bid_price4','bid_size4', 'ask_price4', 'ask_size4',
        #                 'bid_price5', 'bid_size5', 'ask_price5', 'ask_size5','actionday', 'updatetime', 'updatemillisec', 'instrumentid',
        #                 'exchangeid', 'hishigh', 'hislow', 'tradingday', 'presettlement','preclose', 'open', 'upperlimit', 'lowerlimit']
        #         file_1.columns = cols_1
        #         all_data_1 = all_data_1.append(file_1)
    else:
        print(i)
        for j in future_year['trade_date']:
            if i == j:
                contract = future_year[future_year['trade_date'] == i]['contract'].values[0]
                dir_file = '/home/SourceData/{}/{}/{}/{}_{}.csv'.format(exchange,data_path, i, i, contract)
                # print(dir_file)
                file_2 = pd.read_csv(dir_file)
                file_2 = file_2.iloc[:,1:]
                cols_2 = ['tradingday', 'instrumentid', 'exchangeid', 'price','presettlement', 'preclose', 'open', 'high', 'low', 'volume', 'amount',
                        'open_interest', 'upperlimit', 'lowerlimit', 'updatetime','updatemillisec',
                        'bid_price1', 'bid_size1', 'ask_price1', 'ask_size1', 'bid_price2','bid_size2', 'ask_price2', 'ask_size2',
                        'bid_price3', 'bid_size3', 'ask_price3', 'ask_size3', 'bid_price4','bid_size4', 'ask_price4', 'ask_size4',
                        'bid_price5', 'bid_size5', 'ask_price5', 'ask_size5','averageprice','actionday', 'timestamp', 'hishigh', 'hislow']
                file_2.columns = cols_2
                all_data_2 = all_data_2.append(file_2)
#%%
all_data_3 = all_data_2[['timestamp', 'price', 'high', 'low', 'averageprice', 'volume','amount', 'open_interest',
                    'bid_price1', 'bid_size1', 'ask_price1', 'ask_size1', 'bid_price2','bid_size2', 'ask_price2', 'ask_size2',
                    'bid_price3', 'bid_size3', 'ask_price3', 'ask_size3', 'bid_price4','bid_size4', 'ask_price4', 'ask_size4',
                    'bid_price5', 'bid_size5', 'ask_price5', 'ask_size5','actionday', 'updatetime', 'updatemillisec', 'instrumentid',
                    'exchangeid', 'hishigh', 'hislow', 'tradingday', 'presettlement','preclose', 'open', 'upperlimit', 'lowerlimit']]
all_data = pd.concat([all_data_3,all_data_1],axis=0)
all_data['OCC_TIM'] = all_data['actionday'].map(str)+' '+all_data['updatetime'].map(str)+' '+all_data['updatemillisec'].map(str)
all_data['timestamp'] = all_data['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x,'%Y%m%d %H:%M:%S %f')))
all_data['timestamp']=all_data['timestamp']*1000
all_data['timestamp']=all_data['timestamp'].astype(int)
all_data['updatemillisec']=all_data['updatemillisec'].apply(np.int64)
all_data.drop(['OCC_TIM'],axis=1,inplace=True)
del all_data_1, all_data_2, all_data_3
all_data.to_csv('/home/%s/%s/tick/%s_tick_.csv'%(exchange,future,future))


#%%
import pandas as pd
import numpy as np
import os
import datetime
# import xcsc_tushare as ts
import tushare as ts
#
pro = ts.pro_api('591e6891f9287935f45fc712bcf62335a81cd6829ce76c21c0fdf7b2')
# 上期所 SHF; 大商所 DCE; 郑商所 ZCE; 上海能源交易所 INE
future = 'PG'
trading_list = pro.fut_mapping(ts_code='PG.DCE')
year = 2022
exchange = 'dce'
data_path = 'insdata'
#
trading_list['year'] = trading_list['trade_date'].str.extract('(.{0,4})')
trading_list['year'] = trading_list['year'].astype(int)
future_year = trading_list[trading_list['year']>=year]
future_year['contract'] = future_year['mapping_ts_code'].str.extract('(.{0,6})')
future_year['contract'] = future_year['contract'].str.lower()
#
date = []
g = os.walk('/home/SourceData/%s/%s/'%(exchange, data_path))

for path, dir_list, file_list in g:
    for dir_name in dir_list:
        date.append(dir_name)
all_data = pd.DataFrame()
for i in date:
    for j in future_year['trade_date']:
        if i == j:
            contract = future_year[future_year['trade_date'] == i]['contract'].values[0]
            dir_file = '/home/SourceData/{}/{}/{}/{}_{}.csv'.format(exchange,data_path, i, i, contract)
            print(dir_file)
            file = pd.read_csv(dir_file)
            cols =['tradingday','instrumentid','exchangeid','price','presettlement','preclose','open','high','low','volume',
                    'amount','open_interest','upperlimit','lowerLimit','updatetime','updatemillisec',
                    'bid_price1','bid_size1','ask_price1','ask_size1','bid_price2','bid_size2','ask_price2','ask_size2',
                    'bid_price3','bid_size3','ask_price3','ask_size3','bid_price4','bid_size4','ask_price4','ask_size4',
                    'bid_price5','bid_size5','ask_price5','ask_size5','averageprice','actionday','hishigh','hislow','timestamp']
            file.columns = cols
            all_data = all_data.append(file)
all_data.to_csv('/home/%s/%s/tick/%s_tick.csv'%(exchange,future,future))

#%% ricequant data
# shfe
import pandas as pd
import numpy as np
import os
import datetime
# import xcsc_tushare as ts
import tushare as ts
# future = ['AU', 'SN', 'NI', 'SC', 'RB', 'ZN', 'HC', 'FU','CU','AG','BU']
future = ['AG', 'FU']
# future = ['SC']
exchange = 'shfe'
# year = '21'
for f in future:
    print(f)
    date = []
    g = os.walk('/home/xianglake/SH_commodity_data')

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            date.append(dir_name)

    all_data = pd.DataFrame()
    for i in date:
        if i[:2] == '23':
            # print(i)
            dir_file = '/home/xianglake/SH_commodity_data/%s/data_%s_tick_%s.csv'%(i, f,i)
            # print(dir_file)
            file = pd.read_csv(dir_file)
            cols = ['future','datetime', 'trading_date', 'open', 'last', 'high',
                    'low', 'pre_settlement', 'pre_close', 'volume', 'open_interest',
                    'amount', 'limit_up', 'limit_down',
                    'ask_price1', 'ask_price2','ask_price3', 'ask_price4', 'ask_price5',
                    'bid_price1', 'bid_price2','bid_price3', 'bid_price4', 'bid_price5',
                    'ask_size1', 'ask_size2','ask_size3', 'ask_size4', 'ask_size5',
                    'bid_size1', 'bid_size2','bid_size3', 'bid_size4', 'bid_size5',
                    'change_rate']
            file.columns = cols
            all_data = all_data.append(file)
            print(all_data)
    all_data.to_csv('/home/xianglake/songhe/%s/%s/tick/%s_tick_%s.csv'%(exchange,f,f,23))
#%% ricequant dce
import pandas as pd
import numpy as np
import os
import datetime
# import xcsc_tushare as ts
import tushare as ts
future = ['I', 'PG', 'P', 'M', 'JM', 'J', 'I',]
# future = ['FU']
exchange = 'dce'
for f in future:
    print(f)
    date = []
    g = os.walk('/home/xianglake/DCE_commodity_data')

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            date.append(dir_name)

    all_data = pd.DataFrame()
    for i in date:
        dir_file = '/home/xianglake/DCE_commodity_data/%s/data_%s_tick_%s.csv'%(i, f,i)
        # print(dir_file)
        file = pd.read_csv(dir_file)
        cols = ['future','datetime', 'trading_date', 'open', 'last', 'high',
                'low', 'pre_settlement', 'pre_close', 'volume', 'open_interest',
                'amount', 'limit_up', 'limit_down',
                'ask_price1', 'ask_price2','ask_price3', 'ask_price4', 'ask_price5',
                'bid_price1', 'bid_price2','bid_price3', 'bid_price4', 'bid_price5',
                'ask_size1', 'ask_size2','ask_size3', 'ask_size4', 'ask_size5',
                'bid_size1', 'bid_size2','bid_size3', 'bid_size4', 'bid_size5',
                'change_rate']
        file.columns = cols
        all_data = all_data.append(file)
    #
    all_data.to_csv('/home/xianglake/songhe/%s/%s/tick/%s_tick_ricequant.csv'%(exchange,f,f))
#%%
import pandas as pd
pd_day = pd.read_csv('/home/xianglake/songhe/demos/hft_fut_bt/if300_0206_0923_0.003p_0.003l_80_20_50bar_120s_15000000_closes.csv')
pd_day['open_datetime'] = pd.to_datetime(pd_day['opentime'], format='%Y%m%d%H%M%S%f')
pd_day['open_datetime'] = pd_day['open_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
pd_day['close_datetime'] = pd.to_datetime(pd_day['closetime'], format='%Y%m%d%H%M%S%f')
pd_day['close_datetime'] = pd_day['close_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
# pd_day.to_csv('/home/xianglake/songhe/demos/hft_fut_bt/outputs_bt/ru/closes_SHFE.ru.csv')
#%%
from matplotlib import dates as mPlotDATEs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd_day['datetime'] = pd.to_datetime(pd_day['close_datetime'])
x = pd_day['datetime']
x_float = mPlotDATEs.date2num(x)

y = pd_day['totalprofit']

plt.plot(x_float, y)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m%d'))
plt.show()
