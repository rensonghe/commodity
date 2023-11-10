#%%
import pandas as pd
import numpy as np
import os
import tqdm
from tqdm import tqdm
import datetime
from HFT_factor_dce_rice import *
# year = 2022
future = 'IC500'
futuresize = 200
exchange = 'cffex'
min = 120
# dollar = [2_500_000,2_000_000,3_500_000,3_000_000]
dollar = [10_000_000,11_000_000,12_000_000,13_000_000,14_000_000,15_000_000]
# dollar = [10_000_000,11_000_000,12_000_000,13_000_000]
cols = ['future','datetime', 'trading_date', 'open', 'last', 'high',
                'low', 'pre_settlement', 'pre_close', 'volume', 'open_interest',
                'amount', 'limit_up', 'limit_down',
                'ask_price1', 'ask_price2','ask_price3', 'ask_price4', 'ask_price5',
                'bid_price1', 'bid_price2','bid_price3', 'bid_price4', 'bid_price5',
                'ask_size1', 'ask_size2','ask_size3', 'ask_size4', 'ask_size5',
                'bid_size1', 'bid_size2','bid_size3', 'bid_size4', 'bid_size5',
                'change_rate']
all_data = pd.read_csv('/home/data_store/ricequant_data/CFFEX/data_IC_tick_20220101_20231026.csv')
all_data.columns = cols
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
end_time = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

start_time2 = datetime.datetime.strptime('13:00:00', '%H:%M:%S').time()
end_time2 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()



#
data = all_data[(all_data.datetime.dt.time >= start_time) & (all_data.datetime.dt.time <= end_time)|
                 (all_data.datetime.dt.time >= start_time2) & (all_data.datetime.dt.time <= end_time2)
                 ]

data = data.sort_values(by='datetime', ascending=True)
#
# data['time'] = data['datetime'].dt.strftime('%H:%M:%S')
# data.loc[data['time']=='09:00:00', 'size'] = 0
# data.loc[abs(data['size'])>10000, 'size'] = 0

data = data.set_index('datetime')

trade = data.loc[:, ['closetime', 'price', 'volume', 'open_interest','amount', 'size']]
depth = data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1','bid_size1']]
#
start = time.time()

# depth_factor = depth_factor_process(depth, rolling=60)
# trade_factor = trade_factor_process(trade, rolling=60)
final_data = add_factor_process(depth=depth, trade=trade,futuresize=futuresize,min=min)
# order_type_factor = order_type_process(depth=None, trade=trade)
end = time.time()
print('Total Time = %s' % (end - start))

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist
# add_factor = reduce_mem_usage(add_factor)[0]
# order_type_factor = reduce_mem_usage(order_type_factor)[0]

del data, depth, trade

# final_data = pd.merge(add_factor, order_type_factor, on='closetime', how='left')

# del add_factor, order_type_factor

# final_data = final_data.fillna(0)
# final_data = final_data.replace(np.inf, 1)
# final_data = final_data.replace(-np.inf, -1)

# dce
# final_data['vwapv_5s'] = (final_data['price']*final_data['volume']).rolling(2*5).sum()/final_data['volume'].rolling(2*5).sum()
# final_data['vwapv_10s'] = (final_data['price']*final_data['volume']).rolling(2*10).sum()/final_data['volume'].rolling(2*10).sum()
# final_data['vwapv_30s'] = (final_data['price']*final_data['volume']).rolling(2*30).sum()/final_data['volume'].rolling(2*30).sum()
# final_data['vwapv_60s'] = (final_data['price']*final_data['volume']).rolling(2*60).sum()/final_data['volume'].rolling(2*60).sum()
# final_data['vwapv_120s'] = (final_data['price']*final_data['volume']).rolling(2*120).sum()/final_data['volume'].rolling(2*120).sum()
final_data['vwapv_5s'] = (final_data['wap1']*final_data['volume']).rolling(2*5).sum()/final_data['volume'].rolling(2*5).sum()
final_data['vwapv_10s'] = (final_data['wap1']*final_data['volume']).rolling(2*10).sum()/final_data['volume'].rolling(2*10).sum()
final_data['vwapv_30s'] = (final_data['wap1']*final_data['volume']).rolling(2*30).sum()/final_data['volume'].rolling(2*30).sum()
final_data['vwapv_60s'] = (final_data['wap1']*final_data['volume']).rolling(2*60).sum()/final_data['volume'].rolling(2*60).sum()
final_data['vwapv_120s'] = (final_data['wap1']*final_data['volume']).rolling(2*120).sum()/final_data['volume'].rolling(2*120).sum()
# final_data['mid_price'] = (final_data['ask_price1']+final_data['bid_price1'])/2
# final_data['vwapv_60s'] = (final_data['price']*final_data['volume']).rolling(4*60).sum()/final_data['volume'].rolling(4*60).sum()
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
# dollar = [500_000, 550_000,400_000,350_000]
#
for i in dollar:
    # dollar = 2_000_000
    # tick_data = add_factor.copy()
    # tick_data['amount'] = tick_data['amount'].fillna(0)
    data = dollar_bar_df(final_data, 'amount',int(i))
    # data.to_csv('/home/%s/%s/tick_factor/%s_tick_factor_%s_2.csv'%(exchange,future, future, i))
    data.to_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor_%s_%smin_%s.csv' % (exchange, future, future, i,min, 22_23))
# data = data.set_index('datetime')

# data.to_csv('/home/%s/tick_factor/%s_tick_factor_%s.csv'%(future, future, dollar))