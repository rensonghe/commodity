# import akshare as ak
# import pandas_ta as ta
import numpy
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import pandas as pd
import numpy as np
from HFT_factor import *
import datetime
import pandas as pd
import numpy as np
from pandas import Series
underlying_symbols_str='AG'

for month in ('2201', '2202', '2203', '2204', '2205', '2206', '2207', '2208', '2209', '2210', '2211', '2212'):

    label_data01 = pd.read_csv(
        '/home/xianglake/yiliang/data/%s_dollarbar_label_%s_%s.csv' % (underlying_symbols_str, i, month))
    feature_data01 = pd.read_csv(
        '/home/xianglake/yiliang/data/%s_dollarbar_factor_%s_%s.csv' % (underlying_symbols_str, i, month))
    if month == '2201':
        columns1 = label_data01.columns
        columns2 = feature_data01.columns
        label_data = pd.DataFrame(columns=columns1)
        feature_data = pd.DataFrame(columns=columns2)
    label_data = pd.concat([label_data, label_data01])
    feature_data = pd.concat([feature_data, feature_data01])



# '2201','2202','2203','2204','2205','2206','2207',
# '2208','2209','2210','2211','2212'
for month in ('2301','2302','2303','2304','2305','2306','2307'):

    # if month=='2201':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2202':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2203':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2204':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2205':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2206':    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')
    # if month=='2207':    df = pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month),encoding='gbk')
    # if month == '2208':    df = pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month),encoding='gbk')
    # if month == '2209':    df = pd.read_csv(
    #     '/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month, underlying_symbols_str, month),
    #     encoding='gbk')
    # if month == '2210':    df = pd.read_csv(
    #     '/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month, underlying_symbols_str, month),
    #     encoding='gbk')
    # if month == '2211':    df = pd.read_csv(
    #     '/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month, underlying_symbols_str, month),
    #     encoding='gbk')
    # if month == '2212':    df = pd.read_csv(
    #     '/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month, underlying_symbols_str, month),
    #     encoding='gbk')


    df=pd.read_csv('/home/xianglake/SH_commodity_data/{}/data_{}_tick_{}.csv'.format(month,underlying_symbols_str,month), encoding='gbk')

#
# df = pd.concat([df01,df02,df03,df04,df05,df06], axis=0)

    # target
    if underlying_symbols_str=='SC': futuresize=1000
    if underlying_symbols_str=='AG': futuresize=1000
    if underlying_symbols_str=='AU': futuresize=1000
    if underlying_symbols_str=='I': futuresize=100


    # datapre
    def unix_time(dt):
        return time.mktime(dt.timetuple())

    df['datetime']= pd.to_datetime(df['datetime'])
    df['timestamp'] = df['datetime'].apply(unix_time) *1000

    df = df.rename({'b1_v': 'bid_size1','b2_v': 'bid_size2', 'b3_v': 'bid_size3', 'b4_v': 'bid_size4', 'b5_v': 'bid_size5',
                                'a1_v': 'ask_size1', 'a2_v': 'ask_size2', 'a3_v': 'ask_size3', 'a4_v': 'ask_size4', 'a5_v': 'ask_size5',
                                'timestamp':'closetime','last':'price',
                          'a1':'ask_price1','a2':'ask_price2','a3':'ask_price3','a4':'ask_price4','a5':'ask_price5',
                          'b1':'bid_price1','b2':'bid_price2','b3':'bid_price3','b4':'bid_price4','b5':'bid_price5',
                          'total_turnover':'amount'
                          }, axis='columns')

    df['avgprice'] = ((df['amount']-df['amount'].shift(1))/(df['volume']-df['volume'].shift(1))/futuresize).fillna((df['ask_price1'].shift(1)+df['bid_price1'].shift(1))/2)
    df['size'] = np.where((df['avgprice'] > df['ask_price1'].shift(1)), df['volume']-df['volume'].shift(1),np.where((df['avgprice'] < df['bid_price1'].shift(1)),(-1)*(df['volume']-df['volume'].shift(1)),
                                                                                                                    2*(df['avgprice']-(df['ask_price1'].shift(1)+df['bid_price1'].shift(1))/2)/(df['ask_price1'].shift(1)-df['bid_price1'].shift(1))*(df['volume']-df['volume'].shift(1)) ))
    df['price_sig']=(df['size']/(df['volume']-df['volume'].shift(1))).fillna(0)

    df["tho_price"] = (df["ask_price1"] * df["bid_size1"] + df["bid_price1"] * df["ask_size1"]) / (df["bid_size1"] + df["ask_size1"])
    df['mid_price'] = (df['ask_price1']+df['bid_price1'])/2
    df["tho_mean_1m"] = df["tho_price"].rolling(120).mean().shift(-120)
    df["tho_pctg_1m"] = (df["tho_mean_1m"] - df["tho_price"]) / df["tho_price"]
    df["tho_pctg_abs_1m"] = abs(df["tho_pctg_1m"])
    # df["mean2"] = df["mid_price"].rolling(120).mean().shift(-120)

    df["tho_mean_5m"] = df["tho_price"].rolling(600).mean().shift(-600)
    df["tho_pctg_5m"] = (df["tho_mean_5m"] - df["tho_price"]) / df["tho_price"]
    df["tho_pctg_10m"] = (df["tho_mean_5m"] - df["tho_price"]) / df["tho_price"]

    df["tho_price_max_diff"] = df["tho_price"].rolling(120).max().shift(-120)-df["tho_price"]
    df["tho_price_min_diff"] = df["tho_price"].rolling(120).min().shift(-120)-df["tho_price"]
    df["avg_price_min_diff"] = df["avgprice"].rolling(120).min().shift(-120)-df["avgprice"]
    df["avg_price_max_diff"] = df["avgprice"].rolling(120).max().shift(-120)-df["avgprice"]


    df["tho_price_max_diff_5m"] = df["tho_price"].rolling(600).max().shift(-600)-df["tho_price"]
    df["tho_price_min_diff_5m"] = df["tho_price"].rolling(600).min().shift(-600)-df["tho_price"]

    df["tho_price_max_diff_10m"] = df["tho_price"].rolling(1200).max().shift(-1200)-df["tho_price"]
    df["tho_price_min_diff_10m"] = df["tho_price"].rolling(1200).min().shift(-1200)-df["tho_price"]

    df["avg_price_min_diff_5m"] = df["avgprice"].rolling(600).min().shift(-600)-df["avgprice"]
    df["avg_price_max_diff_5m"] = df["avgprice"].rolling(600).max().shift(-600)-df["avgprice"]




    df["tho_price_max_abs_diff_1m"] = abs(df["tho_price_max_diff"])
    df["tho_price_min_abs_diff_1m"] = abs(df["tho_price_min_diff"])
    df["avg_price_max_abs_diff_1m"] = abs(df["avg_price_min_diff"])
    df["avg_price_min_abs_diff_1m"] = abs(df["avg_price_max_diff"])

    df["tho_price_maxmin_abs_1m"] = df[['tho_price_max_abs_diff_1m', 'tho_price_min_abs_diff_1m']].max(axis=1)
    df["tho_mean_30s"] = df["tho_price"].rolling(60).mean().shift(-60)
    df["tho_pctg_30s"] = (df["tho_mean_30s"] - df["tho_price"]) / df["tho_price"]
    df["tho_price_max_diff_30s"] = df["tho_price"].rolling(60).max().shift(-60)-df["tho_price"]
    df["tho_price_min_diff_30s"] = df["tho_price"].rolling(60).min().shift(-60)-df["tho_price"]



    df = df.sort_values(by='datetime', ascending=True)


    start_time = datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()
    end_time = datetime.datetime.strptime('10:15:00','%H:%M:%S').time()

    start_time2 = datetime.datetime.strptime('13:30:00', '%H:%M:%S').time()
    end_time2 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()

    start_time3 = datetime.datetime.strptime('21:00:00', '%H:%M:%S').time()
    end_time3 = datetime.datetime.strptime('23:59:29','%H:%M:%S').time()

    start_time1 = datetime.datetime.strptime('10:30:00', '%H:%M:%S').time()
    end_time1 = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

    start_time4 = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
    end_time4 = datetime.datetime.strptime('02:00:00','%H:%M:%S').time()
    #
    data = df[(df.datetime.dt.time >= start_time) & (df.datetime.dt.time <= end_time)|
                     (df.datetime.dt.time >= start_time1) & (df.datetime.dt.time <= end_time1)|
                     (df.datetime.dt.time >= start_time2) & (df.datetime.dt.time <= end_time2)|
                    (df.datetime.dt.time >= start_time3) & (df.datetime.dt.time <= end_time3)|
                     (df.datetime.dt.time >= start_time4) & (df.datetime.dt.time <= end_time4)]

    data = data.sort_values(by='datetime', ascending=True)

    data = data.set_index('datetime')

    trade = data.loc[:, ['closetime', 'price', 'volume', 'amount', 'open_interest','size','high', 'low']]
    depth = data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
             'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
             'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]


    # feature
    final_data = add_factor_process(depth=depth, trade=trade,futuresize=futuresize)

    data.to_csv("/home/xianglake/yiliang/data/datalabel_{}_{}.csv".format(underlying_symbols_str,month), encoding='gbk')
    final_data.to_csv("/home/xianglake/yiliang/data/datafeature_{}_{}.csv".format(underlying_symbols_str,month), encoding='gbk')


    #
    #
    # label_data=pd.read_csv("./data/0718_test_datalabel_{}_.csv".format(underlying_symbols_str), encoding='gbk')
    # feature_data=pd.read_csv("./data/0718_test_datafeature_{}_.csv".format(underlying_symbols_str), encoding='gbk')
    #
    #
    # s1=[]
    # s2=[]
    # s3=[]
    # s4=[]
    # s5=[]
    # s6=[]
    # s7=[]
    # s8=[]
    # s9=[]
    # s10=[]
    # s11=[]
    #
    # s12=[]
    # s13=[]
    # s14=[]
    # s15=[]
    # s16=[]
    #
    # s17=[]
    #
    #
    # # calc
    # for i in range(8, 93):
    #     corr_tho_pctg_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_pctg_1m'].fillna(0))
    #     corr_tho_max_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_max_diff'].fillna(0))
    #     corr_tho_min_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_min_diff'].fillna(0))
    #     corr_avg_max_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_max_diff'].fillna(0))
    #     corr_avg_min_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_min_diff'].fillna(0))
    #     # print("IC-1m")
    #     # print("corr_tho_pctg_1m " + feature_data.columns[i])
    #     # print(corr_tho_pctg_1m[0, 1])
    #     # print("corr_tho_max_1m " + feature_data.columns[i])
    #     # print(corr_tho_max_1m[0, 1])
    #     # print("corr_tho_min_1m " + feature_data.columns[i])
    #     # print(corr_tho_min_1m[0, 1])
    #     # print("corr_avg_max_1m " + feature_data.columns[i])
    #     # print(corr_avg_max_1m[0, 1])
    #     # print("corr_avg_min_1m " + feature_data.columns[i])
    #     # print(corr_avg_min_1m[0, 1])
    #     s1.append(feature_data.columns[i])
    #     s2.append(corr_tho_pctg_1m[0, 1])
    #     s3.append(corr_tho_max_1m[0, 1])
    #     s4.append(corr_tho_min_1m[0, 1])
    #     s5.append(corr_avg_max_1m[0, 1])
    #     s6.append(corr_avg_min_1m[0, 1])
    #     corr_tho_pctg_5m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_pctg_5m'].fillna(0))
    #     corr_tho_max_5m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_max_diff_5m'].fillna(0))
    #     corr_tho_min_5m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_min_diff_5m'].fillna(0))
    #     corr_avg_max_5m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_max_diff_5m'].fillna(0))
    #     corr_avg_min_5m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_min_diff_5m'].fillna(0))
    #
    #     s7.append(corr_tho_pctg_5m[0, 1])
    #     s8.append(corr_tho_max_5m[0, 1])
    #     s9.append(corr_tho_min_5m[0, 1])
    #     s10.append(corr_avg_max_5m[0, 1])
    #     s11.append(corr_avg_min_5m[0, 1])
    #
    # ###########################################################################################################################
    #
    #     corr_tho_pctg_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_pctg_abs_1m'].fillna(0))
    #     corr_tho_max_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_max_abs_diff_1m'].fillna(0))
    #     corr_tho_min_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_min_abs_diff_1m'].fillna(0))
    #     corr_avg_max_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_max_abs_diff_1m'].fillna(0))
    #     corr_avg_min_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['avg_price_min_abs_diff_1m'].fillna(0))
    #
    #     s12.append(corr_tho_pctg_abs_1m[0, 1])
    #     s13.append(corr_tho_max_abs_1m[0, 1])
    #     s14.append(corr_tho_min_abs_1m[0, 1])
    #     s15.append(corr_avg_max_abs_1m[0, 1])
    #     s16.append(corr_avg_min_abs_1m[0, 1])
    #
    #     corr_tho_maxmin_abs_1m = np.corrcoef(feature_data.iloc[:, i].fillna(0), label_data['tho_price_maxmin_abs_1m'].fillna(0))
    #     s17.append(corr_tho_maxmin_abs_1m[0, 1])
    #
    # #
    # # s1.append('size')
    # # s2.append(np.corrcoef(label_data['size'].fillna(0), label_data['Y_pctg'].fillna(0))[0,1])
    # # s3.append(np.corrcoef(label_data['size'].fillna(0), label_data['tho_price_max_diff'].fillna(0))[0,1])
    # # s4.append(np.corrcoef(label_data['size'].fillna(0), label_data['tho_price_min_diff'].fillna(0))[0,1])
    # # s5.append(np.corrcoef(label_data['size'].fillna(0), label_data['avg_price_max_diff'].fillna(0))[0,1])
    # # s6.append(np.corrcoef(label_data['size'].fillna(0), label_data['avg_price_min_diff'].fillna(0))[0,1])
    # #
    # # s1.append('price_sig')
    # # s2.append(np.corrcoef(label_data['price_sig'].fillna(0), label_data['Y_pctg'].fillna(0))[0,1])
    # # s3.append(np.corrcoef(label_data['price_sig'].fillna(0), label_data['tho_price_max_diff'].fillna(0))[0,1])
    # # s4.append(np.corrcoef(label_data['price_sig'].fillna(0), label_data['tho_price_min_diff'].fillna(0))[0,1])
    # # s5.append(np.corrcoef(label_data['price_sig'].fillna(0), label_data['avg_price_max_diff'].fillna(0))[0,1])
    # # s6.append(np.corrcoef(label_data['price_sig'].fillna(0), label_data['avg_price_min_diff'].fillna(0))[0,1])
    # #
    #
    # s1=Series(s1)
    # s2=Series(s2)
    # s3=Series(s3)
    # s4=Series(s4)
    # s5=Series(s5)
    # s6=Series(s6)
    # s7=Series(s7)
    # s8=Series(s8)
    # s9=Series(s9)
    # s10=Series(s10)
    # s11=Series(s11)
    #
    # s12=Series(s12)
    # s13=Series(s13)
    # s14=Series(s14)
    # s15=Series(s15)
    # s16=Series(s16)
    # s17=Series(s17)
    #
    #
    # corr_m=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17],axis=1)
    #
    # corr_m.columns = ['column_name',
    #                   'tho_pctg_1m', 'tho_price_max_diff', 'tho_price_min_diff', 'avg_price_max_diff','avg_price_min_diff',
    #                   'tho_pctg_5m', 'tho_price_max_diff_5m','tho_price_min_diff_5m', 'avg_price_max_diff_5m','avg_price_min_diff_5m',
    #                   'tho_pctg_abs_1m', 'tho_price_max_abs_diff_1m', 'tho_price_min_abs_diff_1m', 'avg_price_max_abs_diff_1m','avg_price_min_abs_diff_1m',
    #                   'tho_price_maxmin_abs_1m'
    #                   ]
    #
    # print(corr_m)
    # corr_m.to_csv("./log/corr_1m_{}_.csv".format(underlying_symbols_str), encoding='gbk')
    # # print(label_data['Y_pctg'].quantile(0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95)
    #
    # #
    # # label_data['y_1m_new_lable'] = (label_data["tho_price_maxmin_abs_1m"] >= quantile1) * 1 + (label_data["tho_price_maxmin_abs_1m"] <= -quantile1) * (-1) + 0
    #
    #
    #




#
# df["tho_price"] = (df["AskP1"] * df["BidV1"] + df["BidP1"] * df["AskV1"]) / (df["BidV1"] + df["AskV1"])