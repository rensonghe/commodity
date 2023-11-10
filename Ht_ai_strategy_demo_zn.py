# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:15:37 2023

@author: linyiSky
"""
import pandas as pd

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:30:53 2023

@author: linyiSky
"""

import os
import csv

# import datetime

from datetime import datetime, timedelta
from wtpy.WtDataDefs import WtBarRecords, WtTickRecords, WtOrdDtlRecords, WtOrdQueRecords, WtTransRecords
from wtpy import CtaContext, HftContext
# from LogUtils import LogUtils

from HtDataDefs import TickData, BarData, OrderData, TradeData

from CtaTemplate import CtaTemplate

from datetime import datetime
import time
import joblib

from HFT_factor_online import *


# 日志类使用
# logger = LogUtils().get_log()


def makeTime(date: int, time: int, secs: int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date / 10000), month=int(date % 10000 / 100), day=date % 100,
                    hour=int(time / 100), minute=time % 100, second=int(secs / 1000), microsecond=secs % 1000 * 1000)


class Ht_ai_strategy_demo_ag(CtaTemplate):

    symbol = 'ZN'

    split_count = 5

    init_size = 60
    pos_rate = 0.3  # 持仓比例

    futuresize = 5
    pricetick = 5
    slip = 1
    threshold = 4_000_000

    side_long = 0.6955197226631603
    side_short = 0.3417638658162238
    out = 0.5115137965170506  # 第二个模型阈值

    base_path = '/home/wtpy-master/demos/sh_fut_real/'  # 模型路径

    y_pred_side_list = []
    y_pred_out_list = []

    # capital = 1_000_000
    capital = 11000
    deposit_rate = 0.1

    def __init__(self, name: str, code: str, expsecs: int, offset: int, signalfile: str = "", freq: int = 30):
        super().__init__(name, code, expsecs, offset, signalfile, freq)

        self.last_time = int(time.time())
        self.signal_time = int(time.time())
        self.factor_time = int(time.time())
        self.kill_time = 0
        self.single_result = None
        self.strategy_time = int(time.time())
        self.ms_time = int(time.time() * 1000)
        self.print_time = time.time() // 1800

        self.tick_data = []

        self.model_side_0 = joblib.load('{}/model/{}/{}_lightGBM_side_0.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_side_1 = joblib.load('{}/model/{}/{}_lightGBM_side_1.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_side_2 = joblib.load('{}/model/{}/{}_lightGBM_side_2.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_side_3 = joblib.load('{}/model/{}/{}_lightGBM_side_3.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_side_4 = joblib.load('{}/model/{}/{}_lightGBM_side_4.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_out_0 = joblib.load('{}/model/{}/{}_lightGBM_out_0.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_out_1 = joblib.load('{}/model/{}/{}_lightGBM_out_1.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_out_2 = joblib.load('{}/model/{}/{}_lightGBM_out_2.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_out_3 = joblib.load('{}/model/{}/{}_lightGBM_out_3.pkl'.format(self.base_path, self.symbol, self.symbol))
        self.model_out_4 = joblib.load('{}/model/{}/{}_lightGBM_out_4.pkl'.format(self.base_path, self.symbol, self.symbol))

    def on_init_new(self):

        self.__ctx__.stra_get_bars(self.__code__, "m1", 1)
        ticklist = self.__ctx__.stra_get_ticks(self.__code__, 3600)
        if ticklist:
            for i in range(len(ticklist)):
                # print(ticklist['time'][i],ticklist['trading_date'][i],ticklist['action_time'][i],ticklist['price'][i])
                tick_dict = {
                    'closetime': int(datetime.strptime(str(ticklist['time'][i]), '%Y%m%d%H%M%S%f').timestamp() * 1000),
                    'ask_price1': ticklist['ask_price_0'][i], 'ask_size1': ticklist['ask_qty_0'][i],
                    'bid_price1': ticklist['bid_price_0'][i], 'bid_size1': ticklist['bid_qty_0'][i],
                    'ask_price2': ticklist['ask_price_1'][i], 'ask_size2': ticklist['ask_qty_1'][i],
                    'bid_price2': ticklist['bid_price_1'][i], 'bid_size2': ticklist['bid_qty_1'][i],
                    'ask_price3': ticklist['ask_price_2'][i], 'ask_size3': ticklist['ask_qty_2'][i],
                    'bid_price3': ticklist['bid_price_2'][i], 'bid_size3': ticklist['bid_qty_2'][i],
                    'ask_price4': ticklist['ask_price_3'][i], 'ask_size4': ticklist['ask_qty_3'][i],
                    'bid_price4': ticklist['bid_price_3'][i], 'bid_size4': ticklist['bid_qty_3'][i],
                    'ask_price5': ticklist['ask_price_4'][i], 'ask_size5': ticklist['ask_qty_4'][i],
                    'bid_price5': ticklist['bid_price_4'][i], 'bid_size5': ticklist['bid_qty_4'][i],
                    'price': ticklist['price'][i], 'volume': ticklist['total_volume'][i],
                    'amount': ticklist['total_turnover'][i], 'open_interest': ticklist['open_interest'][i]
                }
                self.tick_data.append(tick_dict)

            self.log_text(f"Init load data:{self.__code__},count:{len(ticklist)}")

    def on_tick_new(self, newTick: TickData):

        closetime = newTick.closetime
        tick_dict = {'closetime': newTick.closetime,
                     'ask_price1': newTick.ask_price_1, 'ask_size1': newTick.ask_volume_1,
                     'bid_price1': newTick.bid_price_1, 'bid_size1': newTick.bid_volume_1,
                     'ask_price2': newTick.ask_price_2, 'ask_size2': newTick.ask_volume_2,
                     'bid_price2': newTick.bid_price_2, 'bid_size2': newTick.bid_volume_2,
                     'ask_price3': newTick.ask_price_3, 'ask_size3': newTick.ask_volume_3,
                     'bid_price3': newTick.bid_price_3, 'bid_size3': newTick.bid_volume_3,
                     'ask_price4': newTick.ask_price_4, 'ask_size4': newTick.ask_volume_4,
                     'bid_price4': newTick.bid_price_4, 'bid_size4': newTick.bid_volume_4,
                     'ask_price5': newTick.ask_price_5, 'ask_size5': newTick.ask_volume_5,
                     'bid_price5': newTick.bid_price_5, 'bid_size5': newTick.bid_volume_5,
                     'price': newTick.price, 'volume': newTick.total_volume, 'amount': newTick.total_turnover,
                     'open_interest': newTick.open_interest
                     }
        self.tick_data.append(tick_dict)

        if newTick.closetime:

            if int(newTick.closetime / 1000) - int(self.kill_time / 1000) > 60:
                # 多仓止盈止损
                if self.get_position(self.__code__) > 0 and abs(self.get_position_avgpx(self.__code__)) != 0:

                    pf = float(newTick.price / abs(self.get_position_avgpx(self.__code__))) - 1

                    con1 = 0
                    if pf > 0.05:
                        con1 = 1
                        self.cancel_all(self.__code__)
                        self.kill_time = newTick.closetime
                        self.sell(price=newTick.bid_price_1, volume=self.get_position(self.__code__),
                                  closetime=newTick.closetime)
                        msg_ = f'多头止盈离场---持仓均价{abs(self.get_position_avgpx(self.__code__))}------平仓价格:{newTick.bid_price_1}---size:{self.get_position(self.__code__)}---time:{newTick.closetime}---symbol:{self.__code__}'
                        self.log_text(msg_)
                    elif pf <= -0.009:
                        con1 = 1
                        self.cancel_all(self.__code__)
                        self.kill_time = newTick.closetime
                        self.sell(price=newTick.bid_price_1, volume=self.get_position(self.__code__),
                                  closetime=newTick.closetime)
                        msg_ = f'多头止损离场-----持仓均价{abs(self.get_position_avgpx(self.__code__))}-------平仓价格:{newTick.bid_price_1}---size:{self.get_position(self.__code__)}---time:{newTick.closetime}---symbol:{self.__code__}'
                        self.log_text(msg_)
                    # if con1 == 1:
                    #     # print('-------------离场时间-----------------',
                    #     # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                    #     self.cancel_all(self.__code__)
                    #     self.kill_time = newBar.closetime / 1000
                    #     self.sell(price=newBar.close * (1 - self.place_rate), volume=self.get_position(self.__code__),
                    #               closetime=newBar.time / 1000)

                # 空仓止盈止损
                if self.get_position(self.__code__) < 0 and abs(self.get_position_avgpx(self.__code__)) != 0:
                    # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
                    # self.cancel_all(closetime=bar.closetime / 1000)
                    pf = 1 - float(newTick.price / abs(self.get_position_avgpx(self.__code__)))
                    con1 = 0
                    if pf > 0.05:
                        con1 = 1
                        self.cancel_all(self.__code__)
                        self.kill_time = newTick.closetime
                        self.buy(price=newTick.ask_price_1, volume=-self.get_position(self.__code__),
                                 closetime=newTick.closetime)
                        msg_ = f'空头止盈离场-----持仓均价{abs(self.get_position_avgpx(self.__code__))}------平仓价格:{newTick.ask_price_1}---size:{self.get_position(self.__code__)}---time:{newTick.closetime}---symbol:{self.__code__}'
                        self.log_text(msg_)
                    elif pf <= -0.009:
                        con1 = 1
                        # print('-------------空头止损离场-------------', '品种:',self.model_symbol)

                        self.cancel_all(self.__code__)
                        self.kill_time = newTick.closetime
                        self.buy(price=newTick.ask_price_1, volume=-self.get_position(self.__code__),
                                 closetime=newTick.closetime)
                        msg_ = f'空头止损离场------持仓均价{abs(self.get_position_avgpx(self.__code__))}---平仓价格:{newTick.ask_price_1}---size:{self.get_position(self.__code__)}---time:{newTick.closetime}---symbol:{self.__code__}'
                        self.log_text(msg_)

            time_10 = int(newTick.closetime / 1000)
            interval_time = 60000 * 25  # 提前储存40分钟数据用于计算因子
            if self.tick_data[-1]['closetime'] - self.tick_data[0][
                'closetime'] > interval_time and time_10 - self.last_time > 0.999:
                self.last_time = time_10
                len_tick_data = int(len(self.tick_data) * 0.99)
                diff_time = self.tick_data[-1]['closetime'] - self.tick_data[-len_tick_data]['closetime']
                if diff_time > interval_time:
                    self.tick_data = self.tick_data[-len_tick_data:]

                df_tick_data = pd.DataFrame(self.tick_data)
                all_data = df_tick_data.sort_values(by='closetime', ascending=True)
                all_data['datetime'] = pd.to_datetime(all_data['closetime'] + 28800000, unit='ms')
                # print(all_data)
                # all_data['size'] = np.where((all_data['open_interest'] - all_data['open_interest'].shift(1))>0, all_data['volume']-all_data['volume'].shift(1),np.where((all_data['open_interest'] - all_data['open_interest'].shift(1))<0,(-1)*(all_data['volume']-all_data['volume'].shift(1)),0))

                all_data['avgprice'] = ((all_data['amount'] - all_data['amount'].shift(1)) / (
                        all_data['volume'] - all_data['volume'].shift(1)) / self.futuresize).fillna(
                    (all_data['ask_price1'].shift(1) + all_data['bid_price1'].shift(1)) / 2)
                all_data['size'] = np.where((all_data['avgprice'] > all_data['ask_price1'].shift(1)),
                                            all_data['volume'] - all_data['volume'].shift(1),
                                            np.where((all_data['avgprice'] < all_data['bid_price1'].shift(1)),
                                                     (-1) * (all_data['volume'] - all_data['volume'].shift(1)),
                                                     2 * (all_data['avgprice'] - (
                                                             all_data['ask_price1'].shift(1) + all_data[
                                                         'bid_price1'].shift(1)) / 2) / (
                                                             all_data['ask_price1'].shift(1) - all_data[
                                                         'bid_price1'].shift(1)) * (
                                                             all_data['volume'] - all_data['volume'].shift(1))))

                all_data = all_data.set_index('datetime')
                trade = all_data.loc[:, ['closetime', 'price', 'volume', 'amount', 'open_interest', 'size']]
                depth = all_data.loc[:,
                        ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2',
                         'bid_price2',
                         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
                         'bid_price4',
                         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]

                factor = add_factor_process(depth=depth, trade=trade, futuresize=self.futuresize)
                factor['vwapv_30s'] = (factor['price'] * factor['volume']).rolling(60).sum() / factor['volume'].rolling(
                    60).sum()
                if factor['amount'].iloc[-1] - factor['amount'].iloc[-2] >= self.threshold:
                    print('bar采样触发阈值时间:', newTick.closetime, '品种:', self.__code__)
                    signal = factor.iloc[-1:, :]
                    X_test = np.array(signal.iloc[:, 5:90]).reshape(1, -1)
                    # msg_ = f'factor:{signal.iloc[:, 5:90]}--time:{newTick.closetime}---symbol:{self.__code__}'
                    # self.log_text(msg_)
                    y_pred_side_0 = self.model_side_0.predict(X_test, num_iteration=self.model_side_0.best_iteration)
                    y_pred_side_1 = self.model_side_1.predict(X_test, num_iteration=self.model_side_1.best_iteration)
                    y_pred_side_2 = self.model_side_2.predict(X_test, num_iteration=self.model_side_2.best_iteration)
                    y_pred_side_3 = self.model_side_3.predict(X_test, num_iteration=self.model_side_3.best_iteration)
                    y_pred_side_4 = self.model_side_4.predict(X_test, num_iteration=self.model_side_4.best_iteration)
                    y_pred_side = (y_pred_side_0[0] + y_pred_side_1[0] + y_pred_side_2[0] + y_pred_side_3[0] +
                                   y_pred_side_4[0]) / 5
                    self.y_pred_side_list.append([y_pred_side])
                    msg_ = f'批式方向信号:{self.y_pred_side_list[-1]}--time:{newTick.closetime}---symbol:{self.__code__}'
                    self.log_text(msg_)

                    y_pred_side_df = pd.DataFrame(self.y_pred_side_list, columns=['predict'])

                    if y_pred_side_df['predict'].iloc[-1] >= self.side_long or y_pred_side_df['predict'].iloc[
                        -1] <= self.side_short:
                        y_pred_out_0 = self.model_out_0.predict(X_test, num_iteration=self.model_out_0.best_iteration)
                        y_pred_out_1 = self.model_out_1.predict(X_test, num_iteration=self.model_out_1.best_iteration)
                        y_pred_out_2 = self.model_out_2.predict(X_test, num_iteration=self.model_out_2.best_iteration)
                        y_pred_out_3 = self.model_out_3.predict(X_test, num_iteration=self.model_out_3.best_iteration)
                        y_pred_out_4 = self.model_out_4.predict(X_test, num_iteration=self.model_out_4.best_iteration)
                        y_pred_out = (y_pred_out_0[0] + y_pred_out_1[0] + y_pred_out_2[0] + y_pred_out_3[0] +
                                      y_pred_out_4[0]) / 5
                        self.y_pred_out_list.append([y_pred_out])
                        y_pred_out_df = pd.DataFrame(self.y_pred_out_list, columns=['out'])
                        msg_ = f'入场信号:{self.y_pred_out_list[-1]}-----time:{newTick.closetime}---symbol:{self.__code__}'
                        self.log_text(msg_)

                        # 策略逻辑

                        price = round(factor['vwapv_30s'].iloc[-1] / self.pricetick) * self.pricetick  # 挂单价格

                        # 读取当前持仓
                        curPos = self.get_position(self.__code__)
                        position_value = curPos * price * self.futuresize  # 持仓金额
                        # place_value = (self.capital * self.pos_rate / self.split_count)/self.deposit_rate  # 挂单金额
                        place_value = self.capital / self.deposit_rate  # 挂单金额
                        buy_size = int(round(place_value / (newTick.ask_price_1 * self.futuresize), 8))  # 买单量
                        sell_size = int(round(place_value / (newTick.bid_price_1 * self.futuresize), 8))  # 卖单量
                        # max_limited_order_value = (self.capital * self.pos_rate) / self.deposit_rate  # 最大挂单金额
                        max_limited_order_value = self.capital / self.deposit_rate  # 最大挂单金额

                        # 计算挂单金额
                        limit_orders_values = 0
                        final_values = 0
                        ordict = self.get_active_limit_orders()
                        for key in ordict.keys():
                            limit_orders_values += float(ordict[key].price) * abs(
                                float(ordict[key].leftQty)) * self.futuresize
                        # 持仓金额+挂单金额
                        final_values = limit_orders_values + self.get_position_avgpx(self.__code__) * abs(
                            curPos) * self.futuresize

                        time_1 = datetime.strptime('09:05:00', '%H:%M:%S').time()
                        time_2 = datetime.strptime('13:30:00', '%H:%M:%S').time()
                        time_3 = datetime.strptime('15:30:00', '%H:%M:%S').time()
                        time_4 = datetime.strptime('21:05:00', '%H:%M:%S').time()
                        time_5 = datetime.strptime('23:59:59', '%H:%M:%S').time()
                        time_6 = datetime.strptime('00:00:00', '%H:%M:%S').time()
                        time_7 = datetime.strptime('02:29:59', '%H:%M:%S').time()

                        if (factor.index[-1].time() >= time_1 and factor.index[-1].time() <= time_3) or (
                                factor.index[-1].time() >= time_4 and factor.index[-1].time() <= time_5) or (factor.index[-1].time() >= time_6 and factor.index[-1].time() <= time_7):

                            # 平多仓
                            if float(y_pred_side_df['predict'].iloc[-1]) <= self.side_short and float(
                                    y_pred_out_df['out'].iloc[-1]) >= self.out and curPos > 0:
                                # print('-------------平仓之前撤销所有订单-------------')
                                self.cancel_all(self.__code__)
                                msg_ = f'下空单平多仓撤单'
                                self.log_text(msg_)
                                # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                                self.sell(price=price - (self.pricetick * self.slip), volume=curPos,
                                          closetime=newTick.closetime / 1000)
                                msg_ = f'下空单平多仓-----持仓均价{abs(self.get_position_avgpx(self.__code__))}------平仓价格:{price - (self.pricetick * self.slip)}---size:{curPos}---time:{newTick.closetime}---symbol:{self.__code__}'
                                self.log_text(msg_)
                                time.sleep(0.1)

                            # 平空仓
                            if float(y_pred_side_df['predict'].iloc[-1]) >= self.side_long and float(
                                    y_pred_out_df['out'].iloc[-1]) >= self.out and curPos < 0:
                                # print('-------------平仓之前撤销所有订单-------------')
                                self.cancel_all(self.__code__)
                                msg_ = f'下多单平空仓撤单'
                                self.log_text(msg_)
                                # print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                                self.buy(price=price + (self.pricetick * self.slip), volume=-curPos,
                                         closetime=newTick.closetime / 1000)
                                msg_ = f'下多单平空仓-------持仓均价{abs(self.get_position_avgpx(self.__code__))}---平仓价格:{price + (self.pricetick * self.slip)}---size:{curPos}---time:{newTick.closetime}---symbol:{self.__code__}'
                                self.log_text(msg_)
                                time.sleep(0.1)

                            # 开空仓
                            # if float(y_pred_side_df['predict'].iloc[-1]) <= self.side_short and float(
                            #         y_pred_out_df['out'].iloc[
                            #             -1]) >= self.out and position_value >= -self.pos_rate * self.capital * (
                            #         1 - 1 / self.split_count) / self.deposit_rate:
                            if float(y_pred_side_df['predict'].iloc[-1]) <= self.side_short and float(
                                    y_pred_out_df['out'].iloc[
                                        -1]) >= self.out and position_value >= - self.capital / self.deposit_rate:

                                # print('open short')
                                # print('--------------开空仓----------------', '品种:',self.model_symbol)
                                # if max_limited_order_value <= final_values * 1.0001:
                                #     self.cancel_all(self.__code__)
                                #     msg_ = f'空单超出挂单金额---撤单'
                                #     self.log_text(msg_)
                                self.sell(price=price - (self.pricetick * self.slip), volume=sell_size,
                                          closetime=newTick.closetime / 1000)
                                msg_ = f'开空仓---开仓价格:{price - (self.pricetick * self.slip)}---size:{-sell_size}---time:{newTick.closetime}---symbol:{self.__code__}'
                                self.log_text(msg_)

                            # 开多仓
                            # if float(y_pred_side_df['predict'].iloc[-1]) >= self.side_long and float(
                            #         y_pred_out_df['out'].iloc[
                            #             -1]) >= self.out and position_value <= self.pos_rate * self.capital * (
                            #         1 - 1 / self.split_count)/ self.deposit_rate:
                            if float(y_pred_side_df['predict'].iloc[-1]) >= self.side_long and float(
                                    y_pred_out_df['out'].iloc[
                                        -1]) >= self.out and position_value <= self.capital / self.deposit_rate:
                                # print('open long')
                                # print('--------------开多仓----------------', '品种:',self.model_symbol)
                                # if max_limited_order_value <= final_values * 1.0001:
                                #     self.cancel_all(self.__code__)
                                #     msg_ = f'多单超出挂单金额---撤单'
                                #     self.log_text(msg_)
                                self.buy(price + (self.pricetick * self.slip), volume=buy_size,
                                         closetime=newTick.closetime / 1000)
                                msg_ = f'开多仓---开仓价格:{price + (self.pricetick * self.slip)}---size:{buy_size}---time:{newTick.closetime}---symbol:{self.__code__}'
                                self.log_text(msg_)





                            else:

                                pass

        '''

        for key,value in ordict.items():
            print("----------------------------",self.last_time,key,value.isBuy,value.isCanceled,value.code,value.leftQty)
            pass

        print("on_tick_new--------------",newTick.price)
        '''

        # print("on_tick_new,last_time,curtime--------------",newTick.price,self.last_time,newTick.closetime)

        # 除了__init__外能使用self.log_text
        # self.log_text(f"on_tick_new:{newTick.price},last_time:{self.last_time},curtime:{newTick.closetime}")

        return

    def on_bar_new(self, newBar: BarData):

        return

    def on_order_new(self, order: OrderData):
        return

    def on_trade_new(self, trade: TradeData):
        return

    def on_backtest_end(self, context: CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return












