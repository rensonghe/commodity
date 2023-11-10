import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import scipy.stats as st
#%%  计算这一行基于bid和ask的wap
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap12(df):
    var1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    var2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_wap34(df):
    var1 = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']
    var2 = df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_swap1(df):
    return df['wap1'] - df['wap3']

def calc_swap12(df):
    return df['wap12'] - df['wap34']

def calc_tswap1(df):
    return -df['swap1'].diff()

def calc_tswap12(df):
    return -df['swap12'].diff()

def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    mid = (df['ask_price1'] + df['bid_price1']) / 2
    return (ask - bid) / mid

def calc_tt1(df):
    p1 = df['ask_price1'] * df['ask_size1'] + df['bid_price1'] * df['bid_size1']
    p2 = df['ask_price2'] * df['ask_size2'] + df['bid_price2'] * df['bid_size2']
    return p2 - p1

def calc_price_impact(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    return (df['ask_price1'] - ask)/df['ask_price1'], (df['bid_price1'] - bid)/df['bid_price1']

# Calculate order book slope
def calc_slope(df):
    v0 = (df['bid_size1']+df['ask_size1'])/2
    p0 = (df['bid_price1']+df['ask_price1'])/2
    slope_bid = ((df['bid_size1']/v0)-1)/abs((df['bid_price1']/p0)-1)+(
                (df['bid_size2']/df['bid_size1'])-1)/abs((df['bid_price2']/df['bid_price1'])-1)
    slope_ask = ((df['ask_size1']/v0)-1)/abs((df['ask_price1']/p0)-1)+(
                (df['ask_size2']/df['ask_size1'])-1)/abs((df['ask_price2']/df['ask_price1'])-1)
    return (slope_bid+slope_ask)/2, abs(slope_bid-slope_ask)

# Calculate order book dispersion
def calc_dispersion(df):
    bspread = df['bid_price1'] - df['bid_price2']
    aspread = df['ask_price2'] - df['ask_price1']
    bmid = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price1']
    bmid2 = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price2']
    amid = df['ask_price1'] - (df['bid_price1'] + df['ask_price1'])/2
    amid2 = df['ask_price2'] - (df['bid_price1'] + df['ask_price1'])/2
    bdisp = (df['bid_size1']*bmid + df['bid_size2']*bspread)/(df['bid_size1']+df['bid_size2'])
    bdisp2 = (df['bid_size1']*bmid + df['bid_size2']*bmid2)/(df['bid_size1']+df['bid_size2'])
    adisp = (df['ask_size1']*amid + df['ask_size2']*aspread)/(df['ask_size1']+df['ask_size2'])
    adisp2 = (df['ask_size1']*amid + df['ask_size2']*amid2)/(df['ask_size1']+df['ask_size2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp)/2, (bdisp2 + adisp2)/2

# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
               'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth

#  order flow imbalance
def calc_ofi(df):
    a = df['bid_size1']*np.where(df['bid_price1'].diff()>=0,1,0)
    b = df['bid_size1'].shift()*np.where(df['bid_price1'].diff()<=0,1,0)
    c = df['ask_size1']*np.where(df['ask_price1'].diff()<=0,1,0)
    d = df['ask_size1'].shift()*np.where(df['ask_price1'].diff()>=0,1,0)
    return (a - b - c + d).fillna(0)

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    # return (np.sum(series**4)*series.shape[0]/3)
    return (series.count() / 3) * np.sum(series ** 4)

def reciprocal_transformation(series):
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)

# Calculate the realized absolute variation
def realized_absvar(series):
    return np.sqrt(np.pi/(2*series.count()))*np.sum(np.abs(series))

# Calculate the realized skew
def realized_skew(series):
    return np.sqrt(series.count())*np.sum(series**3)/(realized_volatility(series)**3)

# Calculate the realized kurtosis
def realized_kurtosis(series):
    return series.count()*np.sum(series**4)/(realized_volatility(series)**4)

def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age

def bid_age(depth, rolling=10):
    bp1 = depth['bid_price1']
    bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return bp1_changes

def ask_age(depth, rolling=10):
    ap1 = depth['ask_price1']
    ap1_changes = ap1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return ap1_changes

def inf_ratio(depth, trade, rolling=100):
    quasi = trade.price.diff().abs().rolling(rolling).sum().fillna(10)
    dif = trade.price.diff(rolling).abs().fillna(10)
    return quasi / (dif + quasi)

def depth_price_range(depth, trade, rolling=100):
    return (depth.ask_price1.rolling(rolling).max() / depth.ask_price1.rolling(rolling).min() - 1).fillna(0)

def arrive_rate(depth, trade, rolling=300):
    res = trade['closetime'].diff(rolling).fillna(0) / rolling
    return res

def bp_rank(depth, trade, rolling=100):
    return ((depth.bid_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)

def ap_rank(depth, trade, rolling=100):
    return ((depth.ask_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)

def avg_rank(depth, trade, rolling=100, futuresize=10):
    avg = ((trade['amount'] - trade['amount'].shift(1)) / (
            trade['volume'] - trade['volume'].shift(1)) / futuresize).fillna(
        (depth['ask_price1'].shift(1) + depth['bid_price1'].shift(1)) / 2)
    return ((pd.Series(avg, index=trade.index).rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)

def price_impact(depth, trade, level=5):
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, level + 1):
        ask += depth[f'ask_price{i}'] * depth[f'ask_size{i}']
        bid += depth[f'bid_price{i}'] * depth[f'bid_size{i}']
        ask_v += depth[f'ask_size{i}']
        bid_v += depth[f'bid_size{i}']
    ask /= ask_v
    bid /= bid_v
    return pd.Series(-(depth['ask_price1'] - ask) / depth['ask_price1'] - (depth['bid_price1'] - bid) / depth['bid_price1'], name="price_impact")

def depth_price_skew(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
    return depth[prices].skew(axis=1)

def depth_price_kurt(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
    return depth[prices].kurt(axis=1)

def rolling_return(depth, trade, rolling=100):
    mp = ((depth.ask_price1 + depth.bid_price1) / 2)
    return (mp.diff(rolling) / mp).fillna(0)

def buy_increasing(depth, trade, rolling=100):

    v = trade['size'].copy()
    v[v < 0] = 0
    return np.log1p(((v.rolling(2 * rolling).sum() + 1) / (v.rolling(rolling).sum() + 1)).fillna(1))

def sell_increasing(depth, trade, rolling=100):
    v = trade['size'].copy()
    v[v > 0] = 0
    return np.log1p(((v.rolling(2 * rolling).sum() - 1) / (v.rolling(rolling).sum() - 1)).fillna(1))

def first_location_of_maximum(x):
    max_value = max(x)  # 一个for 循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1

def ask_price_idxmax(depth, trade, rolling=20):
    return depth['ask_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba', raw=True).fillna(0)

def bid_price_idxmax(depth, trade, rolling=20):
    return depth['bid_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba', raw=True).fillna(0)

def avg_price_idxmax(depth, trade, rolling=20, futuresize=10):
    avg = ((trade['amount'] - trade['amount'].shift(1)) / (
                trade['volume'] - trade['volume'].shift(1)) / futuresize).fillna(
        (depth['ask_price1'].shift(1) + depth['bid_price1'].shift(1)) / 2)
    return pd.Series(avg, index=trade.index).rolling(rolling).apply(first_location_of_maximum, engine='numba', raw=True).fillna(0)

def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))

def center_deri_two(depth, trade, rolling=20):
    return depth['ask_price1'].rolling(rolling).apply(mean_second_derivative_centra, engine='numba', raw=True).fillna(0)

def quasi(depth, trade, rolling=100):
    return depth.ask_price1.diff(1).abs().rolling(rolling).sum().fillna(0)

def last_range(depth, trade, rolling=100):
    return trade.price.diff(1).abs().rolling(rolling).sum().fillna(0)

def arrive_rate_2(depth, trade, rolling=100):
    return (trade.closetime.shift(rolling) - trade.closetime).fillna(0)

def avg_trade_volume(depth, trade, rolling=100):
    return (trade['size'][::-1].abs().rolling(rolling).sum().shift(-rolling + 1)).fillna(0)[::-1]

def avg_spread(depth, trade, rolling=200):
    return (depth.ask_price1 - depth.bid_price1).rolling(rolling).mean().fillna(0)

def avg_turnover(depth, trade, rolling=500):
    return depth[['ask_size1', 'ask_size2', 'ask_size3', 'ask_size4','ask_size5','bid_size1', 'bid_size2', 'bid_size3', 'bid_size4', 'bid_size5']].sum(axis=1)

def abs_volume_kurt(depth, trade, rolling=500):
    return trade['size'].abs().rolling(rolling).kurt().fillna(0)

def abs_volume_skew(depth, trade, rolling=500):
    return trade['size'].abs().rolling(rolling).skew().fillna(0)

def volume_kurt(depth, trade, rolling=500):
    return trade['size'].rolling(rolling).kurt().fillna(0)

def volume_skew(depth, trade, rolling=500):
    return trade['size'].rolling(rolling).skew().fillna(0)

def price_kurt(depth, trade, rolling=500):
    return trade.price.rolling(rolling).kurt().fillna(0)

def price_skew(depth, trade, rolling=500):
    return trade.price.rolling(rolling).skew().abs().fillna(0)

def bv_divide_tn(depth, trade, rolling=10):
    bvs = depth.bid_size1 + depth.bid_size2 + depth.bid_size3 + depth.bid_size4 + depth.bid_size5

    def volume(depth, trade, rolling):
        return trade['size'].copy()

    v = volume(depth=depth, trade=trade, rolling=rolling)
    v[v > 0] = 0
    return (v.rolling(rolling).sum() / bvs).fillna(0)

def av_divide_tn(depth, trade, rolling=10):
    avs = depth.ask_size1 + depth.ask_size2 + depth.ask_size3 + depth.ask_size4 + depth.ask_size5

    def volume(depth, trade, n):
        return trade['size'].copy()

    v = volume(depth=depth, trade=trade, n=rolling)
    v[v < 0] = 0
    return (v.rolling(rolling).sum() / avs).fillna(0)

def weighted_price_to_mid(depth, trade, levels=5, alpha=1):
    def get_columns(name, levels):
        return [name + str(i) for i in range(1, levels + 1)]

    avs = depth[get_columns("ask_size", levels)]
    bvs = depth[get_columns("bid_size", levels)]
    aps = depth[get_columns("ask_price", levels)]
    bps = depth[get_columns("bid_price", levels)]
    mp = (depth['ask_price1'] + depth['bid_price1']) / 2
    return (avs.values * aps.values + bvs.values * bps.values).sum(axis=1) / (avs.values + bvs.values).sum(axis=1) - mp

def _bid_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

def _ask_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

def ask_withdraws(depth, trade):
    ob_values = depth.iloc[:,1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows

def bid_withdraws(depth, trade):
    ob_values = depth.iloc[:,1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows

def z_t(trade, depth):
    """初探市场微观结构：指令单薄与指令单流——资金交易策略之四 成交价的对数减去中间价的对数"""
    # data_dic = self.data_dic  # 调用的是属性
    tick_fac_data = np.log(trade['price']) - np.log((depth['bid_price1'] + depth['ask_price1']) / 2)
    return tick_fac_data

def voi(depth,trade):
    """voi订单失衡 Volume Order Imbalance20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    bid_sub_price = depth['bid_price1'] - depth['bid_price1'].shift(1)
    ask_sub_price = depth['ask_price1'] - depth['ask_price1'].shift(1)

    bid_sub_volume = depth['bid_size1'] - depth['bid_size1'].shift(1)
    ask_sub_volume = depth['ask_size1'] - depth['ask_size1'].shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = depth['bid_size1'][bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = depth['ask_size1'][ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']
    return tick_fac_data

def cal_weight_volume(depth):
    """计算加权的盘口挂单量"""
    # data_dic = self.data_dic
    w = [1 - (i - 1) / 5 for i in range(1, 6)]
    w = np.array(w) / sum(w)
    wb = depth['bid_size1'] * w[0] + depth['bid_size2'] * w[1] + depth['bid_size3'] * w[2] + depth['bid_size4'] * w[3] + depth['bid_size5'] * w[4]
    wa = depth['ask_size1'] * w[0] + depth['ask_size2'] * w[1] + depth['ask_size3'] * w[2] + depth['ask_size4'] * w[3] + depth['ask_size5'] * w[4]
    return wb, wa

def oir(depth, trade):
    wb, wa = cal_weight_volume(depth)
    ori = (wb-wa)/(wa+wb)
    return ori

def voi2(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price1'] - depth['bid_price1'].shift(1)
    ask_sub_price = depth['ask_price1'] - depth['ask_price1'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level2(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price2'] - depth['bid_price2'].shift(1)
    ask_sub_price = depth['ask_price2'] - depth['ask_price2'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level3(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price3'] - depth['bid_price3'].shift(1)
    ask_sub_price = depth['ask_price3'] - depth['ask_price3'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level4(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price4'] - depth['bid_price4'].shift(1)
    ask_sub_price = depth['ask_price4'] - depth['ask_price4'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level5(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price5'] - depth['bid_price5'].shift(1)
    ask_sub_price = depth['ask_price5'] - depth['ask_price5'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def mpb(depth, trade, futuresize=10):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = (trade['amount'] / trade['volume'])/futuresize  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = (depth['bid_price1'] + depth['ask_price1']) / 2
    # tick_fac_data = tp - (mid + mid.shift(1)) / 1000 / 2
    tick_fac_data = tp - (mid + mid.shift(1)) / 2
    return tick_fac_data

def mpb_5min(depth, trade, rolling=120*5, futuresize=10):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = (trade['amount'] / trade['volume'])/futuresize  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = (depth['bid_price1'] + depth['ask_price1']) / 2
    # tick_fac_data = tp - (mid + mid.shift(rolling)) / 1000 / 2
    tick_fac_data = tp - (mid + mid.shift(rolling)) / 2
    return tick_fac_data

def mpc(depth, trade, rolling=120*5):
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    mpc = (mid-mid.shift(rolling))/mid.shift(rolling)
    return mpc


def slope(depth):
    """斜率 价差/深度"""
    # data_dic = self.data_dic
    tick_fac_data = (depth['ask_price1'] - depth['bid_price1']) / (depth['ask_size1'] + depth['bid_size1']) * 2
    return tick_fac_data

def positive_ratio(depth, trade,rolling=20 * 3):
    """积极买入成交额占总成交额的比例"""
    # data_dic = self.data_dic
    buy_positive = pd.DataFrame(0, columns=['amount'], index=trade['amount'].index)
    buy_positive['amount'] = trade['amount']
    # buy_positive[trade['price'] >= depth['ask_price1'].shift(1)] = trade['turnover'][trade['price'] >= depth['ask_price1'].shift(1)]
    buy_positive['amount'] = np.where(trade['price']>=depth['ask_price1'].shift(1), buy_positive['amount'], 0)
    tick_fac_data = buy_positive['amount'].rolling(rolling).sum() / trade['amount'].rolling(rolling).sum()
    return tick_fac_data

def positive_buying(depth, trade, rolling = 60):
    '''
    买入情绪因子：根据积极买入和保守买入
    '''
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['volume']-trade['volume'].shift(1), 0)
    caustious_buy = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['volume']-trade['volume'].shift(1), 0)
    bm = pd.Series(positive_buy, index=trade.index).rolling(rolling).sum()/pd.Series(caustious_buy, index=trade.index).rolling(rolling).sum()
    bm_oi = pd.Series(positive_buy, index=trade.index).rolling(rolling).sum()/trade['open_interest']
    return bm.fillna(0), bm_oi.fillna(0)
def positive_selling(depth, trade, rolling = 60):
    '''
    卖出情绪因子：根据积极卖出和保守卖出
    '''
    positive_sell = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['volume']-trade['volume'].shift(1), 0)
    caustious_sell = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['volume']-trade['volume'].shift(1), 0)
    sm = pd.Series(positive_sell, index=trade.index).rolling(rolling).sum()/pd.Series(caustious_sell, index=trade.index).rolling(rolling).sum()
    sm_oi = pd.Series(positive_sell, index=trade.index).rolling(rolling).sum()/trade['open_interest']
    return sm.fillna(0), sm_oi.fillna(0)

def buying_amplification_ratio(depth, trade, rolling):
    '''

    :param depth:
    :param trade:
    :param rolling:
    :return:
    '''
    biding = depth['bid_size1']*depth['bid_price1'] + depth['bid_size2']*depth['bid_price2'] + depth['bid_size3']*depth['bid_price3'] + depth['bid_size4']*depth['bid_price4'] + depth['bid_size5']*depth['bid_price5']
    asking = depth['ask_size1']*depth['ask_price1'] + depth['ask_size2']*depth['ask_price2'] + depth['ask_size3']*depth['ask_price3'] + depth['ask_size4']*depth['ask_price4'] + depth['ask_size5']*depth['ask_price5']
    amplify_biding = np.where(biding>biding.shift(1), biding-biding.shift(1),0)
    amplify_asking = np.where(asking>asking.shift(1), asking-asking.shift(1),0)
    diff = amplify_biding - amplify_asking
    buying_ratio = (pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount']-trade['amount'].shift(rolling))/rolling
    return buying_ratio.fillna(0)

def buying_amount_ratio(depth, trade, rolling):
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    positive_sell = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    diff = positive_buy - positive_sell
    buying_amount_ratio = ((pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount']-trade['amount'].shift(rolling)))/rolling
    return buying_amount_ratio.fillna(0)

def buying_willing(depth, trade, rolling):
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    positive_buy = np.where(trade['price'] >= depth['ask_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    diff = (amplify_biding - amplify_asking) + (positive_buy - positive_sell)
    buying_willing = pd.Series((pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount']-trade['amount'].shift(rolling)))/rolling
    return buying_willing.fillna(0)

def buying_willing_stength(depth, trade, rolling):
    biding = (depth['bid_size1'] + depth['bid_size2'] + depth['bid_size3'] + depth['bid_size4'] + depth['bid_size5'])
    asking = (depth['ask_size1'] + depth['ask_size2'] + depth['ask_size3'] + depth['ask_size4'] + depth['ask_size5'])
    positive_buy = np.where(trade['price'] >= depth['ask_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    diff = (biding - asking) + (positive_buy - positive_sell)
    buying_stength = pd.Series((pd.Series(diff, index=trade.index).rolling(rolling).mean())/(pd.Series(diff, index=trade.index).rolling(rolling).std())).rolling(rolling).std()/rolling
    return buying_stength.fillna(0)

def buying_amount_strength(depth, trade, rolling):
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1),trade['amount']-trade['amount'].shift(1), 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount']-trade['amount'].shift(1), 0)
    diff = positive_buy - positive_sell
    buying_amount_strength = (pd.Series(((pd.Series(diff, index=trade.index).rolling(rolling).mean())/(pd.Series(diff, index=trade.index).rolling(rolling).std()))).rolling(rolling).std())/rolling
    return buying_amount_strength.fillna(0)

def selling_ratio(depth, trade, rolling):
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    diff = amplify_asking - amplify_biding
    # amount = trade['amount'].copy().reset_index(drop=True)
    selling_ratio = (pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount']-trade['amount'].shift(rolling))/rolling
    return selling_ratio.fillna(0)

def large_order_ratio(depth, trade, rolling=120*2):
    mean = (trade['volume'] - trade['volume'].shift(rolling)).rolling(rolling).mean()
    std = (trade['volume'] - trade['volume'].shift(rolling)).rolling(rolling).std()
    large = np.where(np.abs(trade['size'])>(mean+std), trade['amount']-trade['amount'].shift(rolling),0)
    # amount = trade['amount'].copy().reset_index(drop=True)
    ratio = large/(trade['amount']-trade['amount'].shift(rolling))
    large_order_ratio = (pd.Series(ratio, index=trade.index).rolling(rolling).sum())/rolling
    return large_order_ratio.fillna(0)

def buy_order_aggressivenes_level1(depth, trade, rolling=1000,futuresize=10):
    '''
    买单订单侵略性因子 aggressive level1
    '''
    v = trade['size'].copy()
    # p = trade['price'].copy()
    avg = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    # 买家激进程度
    avg[v<0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    buy_price = np.where((avg>=depth['ask_price1'].shift(1))&(v>=depth['ask_size1'].shift(1)), avg,0)
    amount = np.where((avg>=depth['ask_price1'].shift(1))&(v>=depth['ask_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    volume = np.where((avg >= depth['ask_price1'].shift(1)) & (v >= depth['ask_size1'].shift(1)),
                      trade['volume'] - trade['volume'].shift(1), 0)
    buy_oi_agg_ratio = pd.Series(volume, index=trade.index).rolling(rolling).sum()/trade['open_interest']
    buy_amount_agg_ratio = biding.rolling(rolling).sum()/amount
    buy_price_bias =abs(buy_price-mid.shift(rolling))/mid.shift(rolling)
    return buy_price_bias.fillna(0), buy_amount_agg_ratio.fillna(0), buy_oi_agg_ratio.fillna(0)

def buy_order_aggressivenes_level2(depth, trade, rolling=1000,futuresize=10):
    v = trade['size'].copy()
    # p = trade['price'].copy()
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    avg = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)

    # 买家激进程度
    avg[v<0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    buy_price = np.where((avg>=depth['ask_price1'].shift(1))&(v<=depth['ask_size1'].shift(1)), avg,0)
    amount = np.where((avg>=depth['ask_price1'].shift(1))&(v<=depth['ask_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    volume = np.where((avg >= depth['ask_price1'].shift(1)) & (v <= depth['ask_size1'].shift(1)),
                      trade['volume'] - trade['volume'].shift(1), 0)
    buy_oi_agg_ratio = pd.Series(volume, index=trade.index).rolling(rolling).sum() / trade['open_interest']
    buy_amount_agg_ratio = biding.rolling(rolling).sum()/amount
    buy_price_bias =abs(buy_price-mid.shift(rolling))/mid.shift(rolling)
    return buy_price_bias.fillna(0), buy_amount_agg_ratio.fillna(0), buy_oi_agg_ratio.fillna(0)

def sell_order_aggressivenes_level1(depth, trade, rolling=1000,futuresize=10):
    v = trade['size'].copy()
    # p = trade['price'].copy()
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    avg = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)

    # 卖家激进程度
    avg[v>0] = 0

    mid = (depth['ask_price1']+depth['bid_price1'])/2
    sell_price = np.where((avg<=depth['bid_price1'].shift(1))&(abs(v)>=depth['bid_size1'].shift(1)), avg,0)
    amount = np.where((avg<=depth['bid_price1'].shift(1))&(abs(v)>=depth['bid_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    volume = np.where((avg <= depth['bid_price1'].shift(1)) & (abs(v) >= depth['bid_size1'].shift(1)),
                      trade['volume'] - trade['volume'].shift(1), 0)
    sell_oi_agg_ratio = pd.Series(volume, index=trade.index).rolling(rolling).sum() / trade['open_interest']
    sell_amount_agg_ratio = asking.rolling(rolling).sum()/amount
    sell_price_bias = abs(sell_price-mid.shift(rolling))/mid.shift(rolling)
    return sell_price_bias.fillna(0), sell_amount_agg_ratio.fillna(0), sell_oi_agg_ratio.fillna(0)

def sell_order_aggressivenes_level2(depth, trade, rolling=1000,futuresize=10):
    v = trade['size'].copy()
    # p = trade['price'].copy()
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    avg = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)

    # 卖家激进程度
    avg[v>0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    sell_price = np.where((avg<=depth['bid_price1'].shift(1))&(abs(v)<=depth['bid_size1'].shift(1)), avg,0)
    amount = np.where((avg<=depth['bid_price1'].shift(1))&(abs(v)<=depth['bid_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    volume = np.where((avg <= depth['bid_price1'].shift(1)) & (abs(v) <= depth['bid_size1'].shift(1)),
                      trade['volume'] - trade['volume'].shift(1), 0)
    sell_oi_agg_ratio = pd.Series(volume, index=trade.index).rolling(rolling).sum() / trade['open_interest']
    sell_amount_agg_ratio = asking.rolling(rolling).sum()/amount
    sell_price_bias = abs(sell_price-mid.shift(rolling))/mid.shift(rolling)
    return sell_price_bias.fillna(0), sell_amount_agg_ratio.fillna(0), sell_oi_agg_ratio.fillna(0)


def corr_pv(depth, trade, rolling=120):
    '''
    高频量价相关性
    '''
    p = trade['price']/trade['price'].shift(1)
    # corr_pvpm = trade['price'].rolling(rolling).corr(trade['volume']/abs(trade['size']))
    corr_rm = pd.Series(p-1, index=trade.index).rolling(rolling).corr(abs(trade['size']))
    corr_rv = pd.Series(p-1, index=trade.index).rolling(rolling).corr(trade['volume'])
    # corr_rvpm = pd.Series(p-1, index=trade.index).rolling(rolling).corr(trade['volume']/abs(trade['size']))
    oi_ic = trade['price'].rolling(rolling).corr(trade['open_interest'])

    return corr_rm,corr_rv,oi_ic

def flowInRatio(depth, trade, rolling=120):
    flowInRatio = ((trade['volume']-trade['volume'].shift(rolling))*trade['price']*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(1))))/trade['amount']-trade['amount'].shift(rolling)
    # flowInRatio2 = (trade['open_interest']-trade['open_interest'].shift(1))*trade['price']*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(1)))
    return flowInRatio.fillna(0)

def large_order(depth, trade, rolling=120*10):
    '''
    大单买入卖出因子
    '''
    # avg = ((trade['amount'] - trade['amount'].shift(1)) / (trade['volume'] - trade['volume'].shift(1))) / futuresize
    buy = np.where(trade['size']>0, trade['amount']-trade['amount'].shift(1),0)
    sell = np.where(trade['size']<0, trade['amount']-trade['amount'].shift(1),0)
    large_buy = np.where(pd.Series(buy, index=trade.index) > pd.Series(buy, index=trade.index).rolling(rolling).quantile(0.8),pd.Series(buy, index=trade.index),0)
    large_sell = np.where(pd.Series(sell, index=trade.index) > pd.Series(sell, index=trade.index).rolling(rolling).quantile(0.8),pd.Series(sell, index=trade.index),0)
    large_buy_ratio = pd.Series(large_buy, index=trade.index).rolling(rolling).sum()/(pd.Series(buy, index=trade.index).rolling(rolling).sum()+pd.Series(sell, index=trade.index).rolling(rolling).sum())
    large_sell_ratio = pd.Series(large_sell, index=trade.index).rolling(rolling).sum() / (
                pd.Series(buy, index=trade.index).rolling(rolling).sum() + pd.Series(sell, index=trade.index).rolling(rolling).sum())
    return large_sell_ratio.fillna(0),large_buy_ratio.fillna(0)


def price_weighted_pressure(depth, kws):
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 5)

    bench = kws.setdefault("bench_type","MID")

    _ = np.arange(n1, n2 + 1)

    if bench == "MID":
        bench_prices = depth['ask_price1']+depth['bid_price1']
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")

    def unit_calc(bench_price):
        """比结算价高的价单立马成交，权重=0"""

        bid_d = [bench_price / (bench_price - depth["bid_price%s" % s]) for s in _]
        # bid_d = [_.replace(np.inf,0) for _ in bid_d]
        bid_denominator = sum(bid_d)

        bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]

        press_buy = sum([depth["bid_size%s" % (i + 1)] * w for i, w in enumerate(bid_weights)])

        ask_d = [bench_price / (depth['ask_price%s' % s] - bench_price) for s in _]
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        ask_denominator = sum(ask_d)

        ask_weights = [d / ask_denominator for d in ask_d]

        press_sell = sum([depth['ask_size%s' % (i + 1)] * w for i, w in enumerate(ask_weights)])

        return (np.log(press_buy) - np.log(press_sell)).replace([-np.inf, np.inf], np.nan)

    return unit_calc(bench_prices)

def volume_order_imbalance(depth, kws):

    """
    Reference From <Order imbalance Based Strategy in High Frequency Trading>
    :param data:
    :param kws:
    :return:
    """
    drop_first = kws.setdefault("drop_first", True)

    current_bid_price = depth['bid_price1']

    bid_price_diff = current_bid_price - current_bid_price.shift()

    current_bid_vol = depth['bid_size1']

    nan_ = current_bid_vol[current_bid_vol == 0].index

    bvol_diff = current_bid_vol - current_bid_vol.shift()

    bid_increment = np.where(bid_price_diff > 0, current_bid_vol,
                             np.where(bid_price_diff < 0, 0, np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

    current_ask_price = depth['ask_price1']

    ask_price_diff = current_ask_price - current_ask_price.shift()

    current_ask_vol = depth['ask_size1']

    avol_diff = current_ask_vol - current_ask_vol.shift()

    ask_increment = np.where(ask_price_diff < 0, current_ask_vol,
                             np.where(ask_price_diff > 0, 0, np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

    _ = pd.Series(bid_increment - ask_increment, index=depth.index)

    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan

    _.loc[nan_] = np.nan

    return _

def get_mid_price_change(depth, drop_first=True):
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    _ = mid.pct_change()
    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan
    return _

def Open_Close_Percentage(trade, depth, rolling=120*5):
    openInterestChg = trade['open_interest'] - trade['open_interest'].shift(1)
    volumeChg = trade['volume'] - trade['volume'].shift(1)
    closeContract = (volumeChg - openInterestChg)/2
    openContract = volumeChg - closeContract

    return openContract.rolling(rolling).sum()/closeContract.rolling(rolling).sum()

def Open_Interest_Change(trade, depth, rolling=120*5):
    return np.log(trade['open_interest']/trade['open_interest'].shift(rolling))
#
# 博弈因子
def game(depth, trade, rolling, futuresize=10):
    # avg_price = (trade['amount']/trade['volume'])/futuresize
    avg_price = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)
    vol_buy = np.where(avg_price>depth['bid_price1'].shift(1), trade['volume']-trade['volume'].shift(1),0)
    vol_sell = np.where(avg_price<depth['ask_price1'].shift(1), trade['volume']-trade['volume'].shift(1),0)
    game = pd.Series(vol_buy, index=trade.index).rolling(rolling).sum()/pd.Series(vol_sell, index=trade.index).rolling(rolling).sum()
    game_buy_oi = pd.Series(vol_buy, index=trade.index).rolling(rolling).sum() / trade['open_interest']
    game_sell_oi = pd.Series(vol_sell, index=trade.index).rolling(rolling).sum()/trade['open_interest']
    return game.fillna(0), game_buy_oi.fillna(0), game_sell_oi.fillna(0)

# 资金流向因子
# def flow_amount(trade, depth, rolling):
#     flow_amount = (trade['amount'])*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(1))).fillna(method='ffill')
#     factor = pd.Series(flow_amount, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
#     return factor
# 批量成交划分
def multi_active_buying(trade, depth, rolling):
    # 朴素主动占比因子
    active_buying_1 = (trade['amount']-trade['amount'].shift(1))*(st.t.cdf((trade['price']-trade['price'].shift(1))/(trade['price'].rolling(rolling).std()), df=3))
    active_buying = pd.Series(active_buying_1, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
    std = np.std(np.log(trade['price']/trade['price'].shift(1)))
    # t分布主动占比因子
    active_buying_2 = (trade['amount']-trade['amount'].shift(1))*(st.t.cdf((np.log(trade['price']/trade['price'].shift(rolling)))/std, df=3))
    t_active_buying = pd.Series(active_buying_2, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
    # 标准正太分布主动占比因子
    active_buying_3 = (trade['amount']-trade['amount'].shift(1))*(st.norm.cdf((np.log(trade['price']/trade['price'].shift(rolling)))/std))
    norm_active_buying = pd.Series(active_buying_3, index=trade.index).rolling(rolling).sum() / (trade['amount'] - trade['amount'].shift(rolling))
    # 置信正态分布主动占比因子
    active_buying_4 = (trade['amount']-trade['amount'].shift(1)) * (
        st.norm.cdf((np.log(trade['price'] / trade['price'].shift(rolling))) / 0.1 *1.96))
    confi_norm_active_buying = pd.Series(active_buying_4, index=trade.index).rolling(rolling).sum() / (
                trade['amount'] - trade['amount'].shift(rolling))
    return active_buying.fillna(0), t_active_buying.fillna(0), norm_active_buying.fillna(0), confi_norm_active_buying.fillna(0)

def multi_active_selling(trade, depth, rolling):
    # 朴素主动占比因子
    active_buying_1 = (trade['amount']-trade['amount'].shift(1))*(st.t.cdf((trade['price']-trade['price'].shift(1))/(trade['price'].rolling(rolling).std()), df=3))
    active_selling_1 = (trade['amount']-trade['amount'].shift(1))-active_buying_1
    active_selling = pd.Series(active_selling_1, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
    std = np.std(np.log(trade['price']/trade['price'].shift(1)))
    # t分布主动占比因子
    active_buying_2 = (trade['amount']-trade['amount'].shift(1))*(st.t.cdf((np.log(trade['price']/trade['price'].shift(rolling)))/std, df=3))
    active_selling_2 = (trade['amount']-trade['amount'].shift(1))-active_buying_2
    t_active_selling = pd.Series(active_selling_2, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
    # 标准正太分布主动占比因子
    active_buying_3 = (trade['amount']-trade['amount'].shift(1))*(st.norm.cdf((np.log(trade['price']/trade['price'].shift(rolling)))/std))
    active_selling_3 = (trade['amount']-trade['amount'].shift(1))-active_buying_3
    norm_active_selling = pd.Series(active_selling_3, index=trade.index).rolling(rolling).sum() / (trade['amount'] - trade['amount'].shift(rolling))
    # 置信正态分布主动占比因子
    active_buying_4 = (trade['amount']-trade['amount'].shift(1))* (
        st.norm.cdf((np.log(trade['price'] / trade['price'].shift(rolling))) / 0.1 *1.96))
    active_selling_4 = (trade['amount']-trade['amount'].shift(1))-active_buying_4
    confi_norm_active_selling = pd.Series(active_selling_4, index=trade.index).rolling(rolling).sum() / (
                trade['amount'] - trade['amount'].shift(rolling))
    return active_selling.fillna(0), t_active_selling.fillna(0), norm_active_selling.fillna(0), confi_norm_active_selling.fillna(0)

def regret_factor(depth, trade, rolling,futuresize=10):
    # avg_price = (trade['amount'] / trade['volume'])/futuresize
    small_order_threshold = abs(trade['size']).rolling(rolling).quantile(0.25)
    avg_price = ((trade['amount']-trade['amount'].shift(1))/(trade['volume']-trade['volume'].shift(1))/futuresize).fillna((depth['ask_price1'].shift(1)+depth['bid_price1'].shift(1))/2)
    # normal order
    vol_buy = np.where((avg_price > depth['bid_price1'].shift(1))&(avg_price>trade['price']), trade['volume'] - trade['volume'].shift(1), 0)
    price_buy = np.where((avg_price > depth['bid_price1'].shift(1))&(avg_price>trade['price']), avg_price, 0)
    HCVOL = pd.Series(vol_buy, index=trade.index).rolling(rolling).sum()/trade['volume']
    HCP = (pd.Series(price_buy, index=trade.index).rolling(rolling).mean()/trade['price'])-1
    vol_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < trade['price']),
                        trade['volume'] - trade['volume'].shift(1), 0)
    price_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < trade['price']), avg_price, 0)
    LCVOL = pd.Series(vol_sell, index=trade.index).rolling(rolling).sum() / trade['volume']
    LCP = (pd.Series(price_sell, index=trade.index).rolling(rolling).mean() / trade['price']) - 1
    # small order
    small_vol_buy = np.where(
        (avg_price > depth['bid_price1'].shift(1)) & (avg_price > trade['price']) & abs(trade['size']) < abs(
            small_order_threshold),
        trade['volume'] - trade['volume'].shift(1), 0)
    small_price_buy = np.where(
        (avg_price > depth['bid_price1'].shift(1)) & (avg_price > trade['price']) & abs(trade['size']) < abs(
            small_order_threshold), avg_price, 0)
    small_HCVOL = pd.Series(small_vol_buy, index=trade.index).rolling(rolling).sum() / trade['volume']
    small_HCP = (pd.Series(small_price_buy, index=trade.index).rolling(rolling).mean() / trade['price']) - 1
    small_vol_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < trade['price'])& abs(trade['size']) < abs(
            small_order_threshold),
                        trade['volume'] - trade['volume'].shift(1), 0)
    small_price_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < trade['price'])& abs(trade['size']) < abs(
            small_order_threshold), avg_price, 0)
    small_LCVOL = pd.Series(small_vol_sell, index=trade.index).rolling(rolling).sum() / trade['volume']
    small_LCP = (pd.Series(small_price_sell, index=trade.index).rolling(rolling).mean() / trade['price']) - 1

    return HCVOL, HCP, LCVOL, LCP, small_HCVOL, small_HCP, small_LCVOL, small_LCP

def high_low_price_volume_ratio(depth, trade, rolling):
    volume = trade['size'].copy()
    # high_rolling_percentile_80 = trade['price'].rolling(rolling).apply(lambda x: pd.Series(x).quantile(0.8))
    high_rolling_percentile_80 = trade['price'].rolling(rolling).quantile(0.8)
    high_price_volume_80 = np.where(trade['price']> high_rolling_percentile_80, abs(volume), 0)
    high_price_volume_ratio_80 = pd.Series(high_price_volume_80, index=trade.index).rolling(rolling).sum()/(trade['volume']-trade['volume'].shift(rolling))
    high_rolling_percentile_90 = trade['price'].rolling(rolling).quantile(0.9)
    high_price_volume_90 = np.where(trade['price'] > high_rolling_percentile_90, abs(volume), 0)
    high_price_volume_ratio_90 = pd.Series(high_price_volume_90, index=trade.index).rolling(rolling).sum() / (
                trade['volume'] - trade['volume'].shift(rolling))
    low_rolling_percentile_20 = trade['price'].rolling(rolling).quantile(0.2)
    low_price_volume_20 = np.where(trade['price'] < low_rolling_percentile_20, abs(volume), 0)
    low_price_volume_ratio_20 = pd.Series(low_price_volume_20, index=trade.index).rolling(rolling).sum() / (
                trade['volume'] - trade['volume'].shift(rolling))
    low_rolling_percentile_10 = trade['price'].rolling(rolling).quantile(0.1)
    low_price_volume_10 = np.where(trade['price'] < low_rolling_percentile_10, abs(volume), 0)
    low_price_volume_ratio_10 = pd.Series(low_price_volume_10, index=trade.index).rolling(rolling).sum() / (
            trade['volume'] - trade['volume'].shift(rolling))
    return high_price_volume_ratio_80,low_price_volume_ratio_20,high_price_volume_ratio_90,low_price_volume_ratio_10

def hf_trend_str(depth, trade, rolling=120, futuresize=15):
    avg_price = ((trade['amount'] - trade['amount'].shift(1)) / (
            trade['volume'] - trade['volume'].shift(1)) / futuresize).fillna(
        (depth['ask_price1'].shift(1) + depth['bid_price1'].shift(1)) / 2)

    first_avg_price = pd.Series(avg_price, index=trade.index).rolling(rolling).apply(lambda rows: rows[0], engine='numba', raw=True)
    last_avg_price = pd.Series(avg_price, index=trade.index).rolling(rolling).apply(lambda rows: rows[-1], engine='numba', raw=True)
    first_close = trade['price'].rolling(rolling).apply(lambda rows: rows[0], engine='numba', raw=True)
    last_close = trade['price'].rolling(rolling).apply(lambda rows: rows[-1], engine='numba', raw=True)

    diff_close = abs(trade['price']-trade['price'].shift(1))
    diff_avg_price = abs(avg_price-avg_price.shift(1))
    ht_trend_str_close = (last_close - first_close)/pd.Series(diff_close, index=trade.index).rolling(rolling).sum()
    ht_trend_str_avg = (last_avg_price - first_avg_price) / pd.Series(diff_avg_price, index=trade.index).rolling(rolling).sum()

    return ht_trend_str_close, ht_trend_str_avg

# def hf_real_realupstd(depth, trade, rolling=120):
#     time_period = [1, 5, 15, 30]
#     for time in time_period:
#         log_return = np.log(trade['pirce']/trade['price'].shift(time*120))
#         hf_real_upstd = np.std(log_return, ddof=1)
def oi_rank(depth, trade, rolling=120):
    oi_change = trade['open_interest'] - trade['open_interest'].shift(rolling)
    buy_oi = np.where(oi_change >0, oi_change, 0)
    sell_oi = np.where(oi_change < 0, oi_change, 0)
    # N = [0.99, 0.95, 0.9, 0.8]
    # for rank in N:
    buy_oi_1 = pd.Series(buy_oi, index=trade.index).rolling(rolling).quantile(0.99)
    buy_oi_5 = pd.Series(buy_oi, index=trade.index).rolling(rolling).quantile(0.95)
    buy_oi_10 = pd.Series(buy_oi, index=trade.index).rolling(rolling).quantile(0.9)
    buy_oi_20 = pd.Series(buy_oi, index=trade.index).rolling(rolling).quantile(0.8)
    sell_oi_1 = pd.Series(abs(sell_oi), index=trade.index).rolling(rolling).quantile(0.99)
    sell_oi_5 = pd.Series(abs(sell_oi), index=trade.index).rolling(rolling).quantile(0.95)
    sell_oi_10 = pd.Series(abs(sell_oi), index=trade.index).rolling(rolling).quantile(0.9)
    sell_oi_20 = pd.Series(abs(sell_oi), index=trade.index).rolling(rolling).quantile(0.8)
    oi_rank_1 = pd.Series(buy_oi_1-sell_oi_1).rolling(rolling).mean()/pd.Series(buy_oi_1+sell_oi_1).rolling(rolling).mean()
    oi_rank_5 = pd.Series(buy_oi_5 - sell_oi_5).rolling(rolling).mean() / pd.Series(buy_oi_5 + sell_oi_5).rolling(
        rolling).mean()
    oi_rank_10 = pd.Series(buy_oi_10 - sell_oi_10).rolling(rolling).mean() / pd.Series(buy_oi_10 + sell_oi_10).rolling(
        rolling).mean()
    oi_rank_20 = pd.Series(buy_oi_20 - sell_oi_20).rolling(rolling).mean() / pd.Series(buy_oi_20 + sell_oi_20).rolling(
        rolling).mean()

    return oi_rank_1.fillna(0), oi_rank_5.fillna(0), oi_rank_10.fillna(0), oi_rank_20.fillna(0)


def ATR(depth, trade, rolling=120):

    high = trade['price'].rolling(rolling).max()
    low = trade['price'].rolling(rolling).min()
    price_high = abs(high-trade['price'].shift(rolling))
    price_low = abs(low-trade['price'].shift(rolling))
    high_low = high - low
    data_concat = pd.concat([price_high, price_low, high_low], axis=1)
    TR = data_concat.max(axis=1)
    ATR = pd.Series(TR, index=trade.index).rolling(rolling).mean()
# def price_continues_reverse(depth, trade, rolling=100,futuresize=15):
#
#     low = trade['price'].rolling(rolling).min()
#     close = trade['price'].copy()
#     avg_price = ((trade['amount'] - trade['amount'].shift(1)) / (
#                 trade['volume'] - trade['volume'].shift(1)) / futuresize).fillna(
#         (depth['ask_price1'].shift(1) + depth['bid_price1'].shift(1)) / 2)
#     price_ratio = abs(close - low)/ low
#     price_change_20 = np.where((price_ratio>=0.18)&(price_ratio<=0.22),trade['amount']-trade['amount'].shift(1), 0)
#     amount_change_ratio_20 = pd.Series(price_change_20, index=trade.index).rolling(rolling).sum()/(trade['amount']-trade['amount'].shift(rolling))
#     price_change_50 = np.where((price_ratio >= 0.48) & (price_ratio <= 0.52),
#                                trade['amount'] - trade['amount'].shift(1), 0)
#     amount_change_ratio_50 = pd.Series(price_change_50, index=trade.index).rolling(rolling).sum() / (
#                 trade['amount'] - trade['amount'].shift(rolling))
#     return amount_change_ratio_20, amount_change_ratio_50



#%%
def add_factor_process(depth, trade, futuresize=10, min=40):

    df = pd.DataFrame()
    # df = depth.loc[:,
    #      ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
    #       'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
    #       'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]
    df = trade.loc[:,['closetime', 'price','size', 'volume','amount']]
    # df['price'] = trade.loc[:, ['price']]
    # df['volume'] = trade.loc[:, ['volume']]
    # df['size'] = trade.loc[:, ['size']]
    # df['amount'] = trade.loc[:, ['amount']]
    # df['open_interest'] = trade.loc[:, ['open_interest']]
    df['ask_age'] = ask_age(depth=depth, rolling=60)
    df['bid_age'] = bid_age(depth=depth, rolling=60)
    df['inf_ratio'] = inf_ratio(depth=None, trade=trade, rolling=120 * min)
    df['arrive_rate'] = arrive_rate(depth=None, trade=trade, rolling=120 * min)
    df['arrive_rate_2'] = arrive_rate_2(depth=None, trade=trade, rolling=120 * min)
    df['depth_price_range'] = depth_price_range(depth=depth, trade=None)
    df['bp_rank'] = bp_rank(depth=depth, trade=None, rolling=120 * min)
    df['ap_rank'] = ap_rank(depth=depth, trade=None, rolling=120 * min)
    df['avg_rank'] = avg_rank(depth=depth, trade=trade, rolling=120 * min, futuresize=futuresize)
    df['price_impact'] = price_impact(depth=depth, trade=None, level=5)
    df['depth_price_skew'] = depth_price_skew(depth=depth, trade=None)
    df['depth_price_kurt'] = depth_price_kurt(depth=depth, trade=None)
    df['rolling_return'] = rolling_return(depth=depth, trade=None, rolling=120 * min)
    df['buy_increasing'] = buy_increasing(depth=None, trade=trade, rolling=120 * min)
    df['sell_increasing'] = sell_increasing(depth=None, trade=trade, rolling=120 * min)
    df['ask_price_idxmax'] = ask_price_idxmax(depth=depth, trade=None, rolling=120)
    df['bid_price_idxmax'] = bid_price_idxmax(depth=depth, trade=None, rolling=120)
    df['avg_price_idxmax'] = avg_price_idxmax(depth=depth, trade=trade, rolling=120, futuresize=futuresize)
    df['center_deri_two'] = center_deri_two(depth=depth, trade=None, rolling=120)
    df['quasi'] = quasi(depth=depth, trade=None, rolling=120 * min)
    df['last_range'] = last_range(depth=None, trade=trade, rolling=120 * min)
    df['avg_trade_volume'] = avg_trade_volume(depth=depth, trade=trade, rolling=120 * min)
    df['avg_spread'] = avg_spread(depth=depth, trade=None, rolling=120 * min)
    df['avg_turnover'] = avg_turnover(depth=depth, trade=trade, rolling=120 * min)
    df['abs_volume_kurt'] = abs_volume_kurt(depth=None, trade=trade,rolling=120 * min)
    df['abs_volume_skew'] = abs_volume_skew(depth=None, trade=trade, rolling=120 * min)
    df['volume_kurt'] = volume_kurt(depth=None, trade=trade, rolling=120 * min)
    df['volume_skew'] = volume_skew(depth=None, trade=trade, rolling=120 * min)
    df['price_kurt'] = price_kurt(depth=None, trade=trade, rolling=120 * min)
    df['price_skew'] = price_skew(depth=None, trade=trade, rolling=120 * min)
    df['bv_divide_tn'] = bv_divide_tn(depth=depth, trade=trade, rolling=120* min)
    df['av_divide_tn'] = av_divide_tn(depth=depth, trade=trade, rolling=120* min)
    df['weighted_price_to_mid'] = weighted_price_to_mid(depth=depth, trade=None, levels=5, alpha=1)
    df['ask_withdraws'] = ask_withdraws(depth=depth, trade=None)
    df['bid_withdraws'] = bid_withdraws(depth=depth, trade=None)
    df['z_t'] = z_t(trade=trade, depth=depth)
    df['voi'] = voi(trade=trade, depth=depth)
    df['voi2'] = voi2(depth=depth, trade=trade)
    df['voi2_level2'] = voi2_level2(depth=depth,trade=trade)
    df['voi2_level3'] = voi2_level3(depth=depth, trade=trade)
    df['voi2_level4'] = voi2_level4(depth=depth, trade=trade)
    df['voi2_level5'] = voi2_level5(depth=depth, trade=trade)
    df['wa'], df['wb'] = cal_weight_volume(depth=depth)
    df['slope'] = slope(depth=depth)
    df['mpb'] = mpb(depth=depth, trade=trade,futuresize=futuresize)
    df['mpb_min'] = mpb_5min(depth=depth, trade=trade,rolling=120 * min,futuresize=futuresize)
    df['mpc'] = mpc(depth=depth,trade=None)
    df['oir'] = oir(depth=depth,trade=None)
    df['price_weighted_pressure'] = price_weighted_pressure(depth=depth, kws={})
    df['volume_order_imbalance'] = volume_order_imbalance(depth=depth, kws={})
    df['get_mid_price_change'] = get_mid_price_change(depth=depth, drop_first=True)
    df['positive_buying'],df['positive_buying_oi_change'] = positive_buying(depth=depth,trade=trade,rolling=120*min)
    df['positive_selling'],df['positive_selling_oi_change'] = positive_selling(depth=depth,trade=trade,rolling=120*min)
    df['buying_amount_ratio'] = buying_amount_ratio(depth=depth,trade=trade,rolling=120 * min)
    df['selling_ratio'] = selling_ratio(depth=depth, trade=trade, rolling=120 * min)
    df['buying_amount_strength'] = buying_amount_strength(depth=depth,trade=trade,rolling=120 * min)
    df['buying_amount_ratio'] = buying_amount_ratio(depth=depth,trade=trade,rolling=120 * min)
    df['buying_amount_strength'] = buying_amount_strength(depth=depth,trade=trade,rolling=120 * min)
    df['buying_willing'] = buying_willing(depth=depth,trade=trade,rolling=120 * min)
    df['large_order_ratio'] = large_order_ratio(depth=depth,trade=trade, rolling=120)
    df['Open_Close_Percentage_min'] = Open_Close_Percentage(depth=None, trade=trade, rolling=120 * min)
    df['buy_price_bias_level1'], df['buy_amount_agg_ratio_level1'], df['buy_oi_agg_ratio_level1'] = buy_order_aggressivenes_level1(depth=depth,trade=trade,rolling=120 * min,futuresize=futuresize)
    df['buy_price_bias_level2'], df['buy_amount_agg_ratio_level2'], df['buy_oi_agg_ratio_level2'] = buy_order_aggressivenes_level2(depth=depth,trade=trade,rolling=120 * min,futuresize=futuresize)
    df['sell_price_bias_level1'], df['sell_amount_agg_ratio_level1'], df['sell_oi_agg_ratio_level1'] = sell_order_aggressivenes_level1(depth=depth,trade=trade,rolling=120 * min,futuresize=futuresize)
    df['sell_price_bias_level2'], df['sell_amount_agg_ratio_level2'], df['sell_oi_agg_ratio_level2'] = sell_order_aggressivenes_level2(depth=depth,trade=trade,rolling=120 * min,futuresize=futuresize)
    df['Open_Interest_Change'] = Open_Interest_Change(depth=None, trade=trade, rolling=120 * min)
    df['corr_rm'],df['corr_rv'],df['oi_ic'] = corr_pv(depth=None, trade=trade, rolling=120 * min)
    df['flowInRatio'] = flowInRatio(depth=None, trade=trade, rolling=120 * min)
    df['large_buy_ratio'], df['large_sell_ratio'] = large_order(depth=None, trade=trade, rolling=120 * min)
    df['game'], df['game_buy_oi'], df['game_sell_oi'] = game(depth=depth, trade=trade, rolling=120 * min,futuresize=futuresize)
    df['active_buying'], df['t_active_buying'], df['norm_active_buying'], df['confi_norm_active_buying'] = multi_active_buying(depth=depth, trade=trade, rolling=120 * min)
    df['active_selling'], df['t_active_selling'], df['norm_active_selling'], df['confi_norm_active_selling'] = multi_active_selling(depth=depth, trade=trade, rolling=120 * min)
    df['HCVOL'], df['HCP'], df['LCVOL'], df['LCP'], df['small_HCVOL'], df['small_HCP'], df['small_LCVOL'], df['small_LCP'] = regret_factor(depth=depth, trade=trade, rolling=120 * min,futuresize=futuresize)
    df['high_price_volume_ratio_80'],df['low_price_volume_ratio_20'],df['high_price_volume_ratio_90'],df['low_price_volume_ratio_10'] = high_low_price_volume_ratio(depth=depth, trade=trade, rolling=120*min)
    df['ht_trend_str_close'], df['ht_trend_str_avg'] = hf_trend_str(depth=depth, trade=trade, rolling=120*min, futuresize=futuresize)
    df['oi_rank_1'], df['oi_rank_5'], df['oi_rank_10'], df['oi_rank_20'] = oi_rank(depth=depth, trade=trade, rolling=120*min)

    return df



