#%%
import pandas as pd
import numpy as np
import os
import tqdm
from tqdm import tqdm
import datetime
from HFT_factor_4 import *
# from HFT_factor_3 import *
# year = 2022
future = 'ZN'
futuresize = 5
exchange = 'shfe'
min = 100
year = 22
dollar = [3500000,3000000,4000000,4500000,5000000]
# dollar = [12000000, 13000000, 11000000, 10000000,14000000]
# dollar = [14000000]
# all_data = pd.read_csv('/home/xianglake/songhe/%s/%s/tick/%s_tick.csv'%(exchange,future,future))
all_data = pd.read_csv('/home/xianglake/songhe/%s/%s/tick/%s_tick_%s.csv'%(exchange,future,future, year))
#
all_data['date_time']= pd.to_datetime(all_data['datetime']).dt.tz_localize('Asia/Shanghai')
all_data['datetime']= pd.to_datetime(all_data['datetime'])
# all_data['datetime']= pd.to_datetime(all_data['timestamp']+28800000, unit='ms')

all_data = all_data.iloc[:,1:]
all_data = all_data.sort_values(by='date_time', ascending=True)
# all_data = all_data.reset_index()
# all_data = all_data.rename({'index':'flag'}, axis='columns')
#
# def unix_time(dt):
#     return time.mktime(dt.timetuple())
# all_data['timestamp'] = all_data['datetime'].apply(unix_time) *1000
#
all_data['timestamp'] = all_data['date_time'].apply(lambda x: pd.Timestamp(x).timestamp())
all_data['timestamp'] = (all_data['timestamp'] * 1000).astype(int)
# all_data = all_data.sort_values(by='datetime', ascending=True)
#
# all_data = all_data.reset_index(drop=True)
# all_data = all_data[all_data.datetime>='2022-01-01']
#
# all_data = all_data.rename({'bid_volume1': 'bid_size1','bid_volume2': 'bid_size2', 'bid_volume3': 'bid_size3', 'bid_volume4': 'bid_size4', 'bid_volume5': 'bid_size5',
#                             'ask_volume1': 'ask_size1', 'ask_volume2': 'ask_size2', 'ask_volume3': 'ask_size3', 'ask_volume4': 'ask_size4', 'ask_volume5': 'ask_size5',
#                             'datetime_nano':'closetime','last_price':'price'}, axis='columns')
all_data = all_data.rename({'timestamp':'closetime','last':'price'}, axis='columns')
# all_data['size'] = np.where((all_data['open_interest'] - all_data['open_interest'].shift(1))>0, all_data['volume']-all_data['volume'].shift(1),np.where((all_data['open_interest'] - all_data['open_interest'].shift(1))<0,(-1)*(all_data['volume']-all_data['volume'].shift(1)),0))
#
all_data['avgprice'] = ((all_data['amount']-all_data['amount'].shift(1))/(all_data['volume']-all_data['volume'].shift(1))/futuresize).fillna((all_data['ask_price1'].shift(1)+all_data['bid_price1'].shift(1))/2)
all_data['size'] = np.where((all_data['avgprice'] > all_data['ask_price1'].shift(1)), all_data['volume']-all_data['volume'].shift(1),
                   np.where((all_data['avgprice'] < all_data['bid_price1'].shift(1)),(-1)*(all_data['volume']-all_data['volume'].shift(1)),
                2*(all_data['avgprice']-(all_data['ask_price1'].shift(1)+all_data['bid_price1'].shift(1))/2)/(all_data['ask_price1'].shift(1)-all_data['bid_price1'].shift(1))*(all_data['volume']-all_data['volume'].shift(1)) ))
#
start_time = datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()
end_time = datetime.datetime.strptime('10:15:00','%H:%M:%S').time()

start_time2 = datetime.datetime.strptime('13:30:00', '%H:%M:%S').time()
end_time2 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()

start_time3 = datetime.datetime.strptime('21:00:00', '%H:%M:%S').time()
end_time3 = datetime.datetime.strptime('23:59:29','%H:%M:%S').time()

start_time1 = datetime.datetime.strptime('10:30:00', '%H:%M:%S').time()
end_time1 = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

start_time4 = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
end_time4 = datetime.datetime.strptime('02:30:00','%H:%M:%S').time()
#
data = all_data[(all_data.datetime.dt.time >= start_time) & (all_data.datetime.dt.time <= end_time)|
                 (all_data.datetime.dt.time >= start_time1) & (all_data.datetime.dt.time <= end_time1)|
                 (all_data.datetime.dt.time >= start_time2) & (all_data.datetime.dt.time <= end_time2)|
                (all_data.datetime.dt.time >= start_time3) & (all_data.datetime.dt.time <= end_time3)|
                 (all_data.datetime.dt.time >= start_time4) & (all_data.datetime.dt.time <= end_time4)]

data = data.sort_values(by='datetime', ascending=True)
#
# data['time'] = data['datetime'].dt.strftime('%H:%M:%S')
# data.loc[data['time']=='09:00:00', 'size'] = 0
# data.loc[abs(data['size'])>100000, 'size'] = 0
#
data = data.set_index('datetime')
#
trade = data.loc[:, ['closetime', 'price', 'volume', 'amount', 'open_interest','size']]
depth = data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]
#
# trade = trade.astype('float64')
# depth = depth.astype('float64')

start = time.time()

# depth_factor = depth_factor_process(depth, rolling=60)
# trade_factor = trade_factor_process(trade, rolling=60)
final_data = add_factor_process(depth=depth, trade=trade,futuresize=futuresize, min=min)
# order_type_factor = order_type_process(depth=None, trade=trade)
end = time.time()
print('Total Time = %s' % (end - start))


del data, depth, trade

# final_data = pd.merge(add_factor, order_type_factor, on='closetime', how='left')

# del add_factor, order_type_factor

# final_data = final_data.fillna(0)
# final_data = final_data.replace(np.inf, 1)
# final_data = final_data.replace(-np.inf, -1)
#
# shfe
# final_data['vwapv_120s'] = (final_data['price']*final_data['volume']).rolling(240).sum()/final_data['volume'].rolling(240).sum()
# final_data['vwapv_60s'] = (final_data['price']*final_data['volume']).rolling(120).sum()/final_data['volume'].rolling(120).sum()
# final_data['vwapv_30s'] = (final_data['price']*final_data['volume']).rolling(60).sum()/final_data['volume'].rolling(60).sum()
# final_data['vwapv_2s'] = (final_data['price']*final_data['volume']).rolling(4).sum()/final_data['volume'].rolling(4).sum()
# final_data['vwapv_5s'] = (final_data['price']*final_data['volume']).rolling(10).sum()/final_data['volume'].rolling(10).sum()
# final_data['vwapv_10s'] = (final_data['price']*final_data['volume']).rolling(20).sum()/final_data['volume'].rolling(20).sum()

final_data['vwapv_5s'] = (final_data['wap1']*final_data['volume']).rolling(2*5).sum()/final_data['volume'].rolling(2*5).sum()
final_data['vwapv_10s'] = (final_data['wap1']*final_data['volume']).rolling(2*10).sum()/final_data['volume'].rolling(2*10).sum()
final_data['vwapv_30s'] = (final_data['wap1']*final_data['volume']).rolling(2*30).sum()/final_data['volume'].rolling(2*30).sum()
final_data['vwapv_60s'] = (final_data['wap1']*final_data['volume']).rolling(2*60).sum()/final_data['volume'].rolling(2*60).sum()
final_data['vwapv_120s'] = (final_data['wap1']*final_data['volume']).rolling(2*120).sum()/final_data['volume'].rolling(2*120).sum()

def vwap(df, second, furturesize=futuresize):

    df = (df['amount'].shift(-(second-1)) - df['amount'])/(df['volume'].shift(-(second-1))-df['volume'])/furturesize
    return df

def twap_60s(df):
    df = (df['wap1'].shift(-(1*2-1))+ df['wap1'].shift(-(10*2-1)) +  df['wap1'].shift(-(30*2-1)) +  df['wap1'].shift(-(60*2-1)))/4
    return df
def twap_120s(df):
    df = (df['wap1'].shift(-(1*2-1))+ df['wap1'].shift(-(30*2-1)) +  df['wap1'].shift(-(60*2-1))+ df['wap1'].shift(-(90*2-1))+df['wap1'].shift(-(120*2-1)))/5
    return df
def twap_300s(df):
    df = (df['wap1'].shift(-(1*2-1)) + df['wap1'].shift(-(60*2-1))+ df['wap1'].shift(-(120*2-1))+ df['wap1'].shift(-(240*2-1))+df['wap1'].shift(-(300*2-1)))/5
    return df
def twap_600s(df):
    df = (df['wap1'].shift(-(1*2-1))+df['wap1'].shift(-(120*2-1)) +  df['wap1'].shift(-(300*2-1)) +  df['wap1'].shift(-(480*2-1))+ df['wap1'].shift(-(600*2-1)))/5
    return df

# final_data['vwap_2s_f'] = vwap(final_data, second=2*2)
# final_data['vwap_5s_f'] = vwap(final_data, second=5*2)
# final_data['vwap_10s_f'] = vwap(final_data, second=10*2)
# final_data['vwap_30s_f'] = vwap(final_data, second=30*2)
final_data['twap_60s_f'] = twap_60s(final_data)
final_data['twap_120s_f'] = twap_120s(final_data)
final_data['twap_300s_f'] = twap_300s(final_data)
final_data['twap_600s_f'] = twap_600s(final_data)
# final_data['vwap_120s_f'] = vwap(final_data, second=120*2)

# final_data = final_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply('last')
# time bar
# final_data_price = final_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg({'closetime': 'last','vwap':'last','price':'last','size':'last','highest':'last', 'lowest':'last'})
# final_data = final_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply('mean')
# final_data = final_data.drop(final_data.columns[[0,1,2,3,4,83]],axis=1)
# final_data = pd.merge(final_data, final_data_price, on='datetime', how='left')
#
# final_data = final_data.dropna(axis=0, how='any')
# dollar/volume bar
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        dv_column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column].astype(int)
    ts = 0
    idx = []
    # for i, x in enumerate(tqdm(t)):
    for i in tqdm(range(1, len(t))):
        if t[i] - t[i-1] >= m:
            # print(t[i])
            idx.append(i)
            continue
        # ts += x
        # if ts >= m:
        #     idx.append(i)
        #     ts = 0
        #     continue
    return idx

def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    return df.iloc[idx].drop_duplicates()
#
#
# dollar = [4_000_000]
#
for i in dollar:
    # dollar = 2_000_000
    # tick_data = add_factor.copy()
    # tick_data['amount'] = tick_data['amount'].fillna(0)
    data = dollar_bar_df(final_data, 'amount',int(i))
    # data.to_csv('/home/%s/%s/tick_factor/%s_tick_factor_%s_2.csv'%(exchange,future, future, i))
    data.to_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor4_%s_%smin_%s.csv' % (exchange, future, future, i,min, year))
# data = data.set_index('datetime')

# data.to_csv('/home/%s/tick_factor/%s_tick_factor_%s.csv'%(future, future, dollar))