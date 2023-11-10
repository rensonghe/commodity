# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:30:53 2023

@author: linyiSky
"""

import os
import csv 


from functools import singledispatch
from datetime import datetime, timedelta
from wtpy.WtDataDefs import WtBarRecords, WtTickRecords, WtOrdDtlRecords, WtOrdQueRecords, WtTransRecords
from wtpy import BaseHftStrategy
from wtpy import CtaContext,HftContext
#from LogUtils import LogUtils

from HtDataDefs import TickData,BarData,OrderData,TradeData

from datetime import datetime

#日志类使用
#logger = LogUtils().get_log()


def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)


class CtaTemplate(BaseHftStrategy):

    def __init__(self, name:str, code:str, expsecs:int, offset:int, signal_file:str="",freq:int=30):
        BaseHftStrategy.__init__(self, name)

        '''交易参数'''
        self.__code__ = code            #交易合约
        self.__expsecs__ = expsecs      #订单超时秒数
        self.__offset__ = offset        #指令价格偏移
        self.__freq__ = freq            #交易频率控制，指定时间内限制信号数，单位秒

        '''内部数据'''
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪

        self.signal_data = {}           #加载信号文件
        with open(signal_file, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            for row in csvreader:
                timestamp_ms = int(row[1])   # 13位时间戳
                data_time = datetime.fromtimestamp(int(timestamp_ms) / 1000)
                time = data_time.strftime('%Y%m%d%H%M%S')
                time_ms = int(timestamp_ms % 1000)
                time_str = str(time) + str(time_ms).zfill(3)  # 注意补0
                signal_time = int(time_str)
                self.signal_data.setdefault(signal_time, row)
                

        print('加载完成的信号量data={}'.format(self.signal_data))
        
        self.orders = dict()
        self.orders_info= dict()

        '''
        self.orders_info=self.load_order_info()
        self.orders = dict()        #策略相关的订单
        
        
        for key,value in self.orders_info.items():
            if value.isCanceled or int(float(value.leftQty)) == 0:
                pass
            else:
                self.__orders__[int(value.localid)]=int(value.localid)
                self.orders[int(value.localid)]=int(value.localid)
                
        '''
        
        '''
        for key,value in self.__orders_info.items():
            if bool(value.isCanceled) or int(value.leftQty) == 0:
                pass
            else:
                self.__orders__[int(value.localid)]=int(value.localid)
                
            
        for key,value in self.__orders__.items():
            print(key,value)
            
        '''
        
        
    def on_init_new(self):
        return 
    
    def on_tick_new(self, newTick:TickData):
        return
    
    def on_bar_new(self, newBar:BarData):
        return
    
    def on_order_new(self, order: OrderData):
        return
    
    def on_trade_new(self, trade: TradeData):
        return
    
    
    def log_text(self, logtxt:str):
        self.__ctx__.stra_log_text(logtxt)
        return


    def save_data(self, key:str, val):
        self.__ctx__.user_save_data(key,val)
    
    def load_data(self, key:str,vType = float):
        '''
        读取用户数据
        @key    数据id
        @defVal 默认数据,如果找不到则返回改数据,默认为None
        @return 返回值,默认处理为float数据
        '''
        return self.__ctx__.user_load_data(key,None,vType)
    
    
    def on_init(self, context:HftContext):
        '''
        策略初始化，启动的时候调用\n
        用于加载自定义数据\n
        @context    策略运行上下文
        '''
        print ("-------------------------------------------------------")
        
        #加载数据
        self.__ctx__ = context
        self.commInfo = context.stra_get_comminfo(self.__code__)
        # self.__ctx__.stra_get_ticks(self.__code__, self.__bar_cnt__)
        #self.__ctx__.stra_log_text(f"on_init:{self.__bar_cnt__}")
        
        #先订阅实时数据
        context.stra_sub_ticks(self.__code__)
        self.on_init_new()
        
    def on_backtest_end(self, context:CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return
        
    def on_tick(self, context:HftContext, stdCode:str, newTick:dict):
        
        td=TickData(newTick)
        self.on_tick_new(td)
        return
    
    def on_bar(self, context:HftContext, stdCode:str, period:str, newBar:dict):
        bd=BarData(period,newBar)
        self.on_bar_new(bd)
        return

    def on_channel_ready(self, context:HftContext):
        context.stra_log_text("交易通道准备完成")
        self.__channel_ready__ = True

    def on_channel_lost(self, context:HftContext):
        context.stra_log_text("交易通道连接丢失")
        self.__channel_ready__ = False

    def on_entrust(self, context:HftContext, localid:int, stdCode:str, bSucc:bool, msg:str, userTag:str):
        if bSucc:
            context.stra_log_text("%s下单成功，本地单号：%d" % (stdCode, localid))
        else:
            context.stra_log_text("%s下单失败，本地单号：%d，错误信息：%s" % (stdCode, localid, msg))

    def on_order(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        
        #context.user_save_data(localid,"active")
        #self.__ctx__.user_save_data(localid,"active")
        if localid not in self.__orders__:
            return

        if isCanceled or leftQty == 0:
            self.__orders__.pop(localid)
            #self.__ctx__.stra_log_text("cancel localid:%d" % (localid))
            #self.__ctx__.user_save_data(localid,"unactive")
            
            if self.__cancel_cnt__ > 0:
                self.__cancel_cnt__ -= 1
                self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))
                
        od=OrderData(localid,stdCode,isBuy,totalQty,leftQty,price,isCanceled,userTag)
        self.orders_info[localid]=od
        self.save_order_info(od)
        self.on_order_new(od)
        
        return
    
    def get_active_limit_orders(self):
        
        ret=dict()
        for key,value in self.orders_info.items():
            
            if value.isCanceled or int(float(value.leftQty)) == 0:
                #print("----------------------------",key,bool(value.isBuy),value.isCanceled,value.code,value.leftQty)
                pass
            else:
                #print("----------------------------",key,bool(value.isBuy),value.isCanceled,value.code,value.leftQty)
                ret[int(value.localid)]=value

        return ret
    

    def on_trade(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        
        td=TradeData(localid,stdCode,isBuy,qty,price,userTag)
        self.on_trade_new(td)
    
        return


    def buy(self,price:float,volume:float,closetime:int):
        ids=self.__ctx__.stra_buy(self.__code__, price=price,qty=volume,userTag="buy",flag=0)
        #将订单号加入到管理中
        for localid in ids:
            self.__orders__[localid] = localid
            #self.__ctx__.user_save_data(localid,"active")
            
        self.__last_entry_time__ = closetime
        #self.__last_entry_time__ = now
        
    def sell(self,price:float,volume:float,closetime:int):
        ids=self.__ctx__.stra_sell(self.__code__, price=price,qty=volume,userTag="sell",flag=0)
        #将订单号加入到管理中
        for localid in ids:
            self.__orders__[localid] = localid
            #self.__ctx__.user_save_data(localid,"active")
            
        self.__last_entry_time__ = closetime
        
    def get_ticks(self,code, count) -> WtTickRecords:
        return self.__ctx__.stra_get_ticks(code, count)
    
    def get_position(self,code):
        return self.__ctx__.stra_get_position(code)
    
    def get_position_avgpx(self, stdCode:str = ""):
        return self.__ctx__.stra_get_position_avgpx(stdCode)
    
    
    def cancel_all(self,code):
        undone = self.__ctx__.stra_get_undone(self.__code__)
        if undone != 0:
            isBuy = (undone > 0)
            ids = self.__ctx__.stra_cancel_all(self.__code__, isBuy)
            for localid in ids:
                self.__orders__[localid] = localid
            self.__cancel_cnt__ += len(ids)
            self.__ctx__.stra_log_text("cancelcnt -> %d" % (self.__cancel_cnt__))
        self.__channel_ready__ = True
        
    def cancel_local_all(self,code):
        for localid in self.__orders__:
            self.__ctx__.stra_cancel(localid)
            self.__cancel_cnt__ += 1
            self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))
            
            
        
    def check_orders(self):
        #如果未完成订单不为空
        if len(self.__orders__.keys()) > 0 and self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            span = now - self.__last_entry_time__
            if span.total_seconds() > self.__expsecs__: #如果订单超时，则需要撤单
                for localid in self.__orders__:
                    self.__ctx__.stra_cancel(localid)
                    self.__cancel_cnt__ += 1
                    self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))
                    

    def save_order_info(self,ordata:OrderData):
        
        currrent_path = os.path.dirname(__file__)
        strTradingday=GetCurTradingDay()
        orderfile_path = os.path.join(currrent_path,strTradingday+'_order.csv')
        
        #print(orderfile_path)
        #ordata=OrderData(123456,'rb2310',True,45,5,4506,False,'tag')
        
        #logger.info(f"save_order path:{orderfile_path}")
       # def __init__(self,localid:int, stdCode:str, 
       #              isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        
        # 检查文件是否存在
        if not os.path.isfile(orderfile_path):
            # 如果文件不存在，创建一个新的文件，并写入标题和数据
            with open(orderfile_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['本地编号', '合约ID', 'isBuy','总成交量','剩余量','价格','isCanceled','userTag'])  # 写入标题
                #tmpdata={ordata.localid,ordata.code,ordata.isBuy,ordata.totalQty,ordata.leftQty,ordata.price,ordata.isCanceled,ordata.userTag}
                tmpdata=[ordata.localid,ordata.code,ordata.isBuy,ordata.totalQty,ordata.leftQty,ordata.price,ordata.isCanceled,ordata.userTag]
                writer.writerow(tmpdata)  # 写入数据
        else:
            # 如果文件存在，追加数据
            with open(orderfile_path, 'a', newline='') as file:
                writer = csv.writer(file)
                #tmpdata={ordata.localid,ordata.code,ordata.isBuy,ordata.totalQty,ordata.leftQty,ordata.price,ordata.isCanceled,ordata.userTag}
                tmpdata=[ordata.localid,ordata.code,ordata.isBuy,ordata.totalQty,ordata.leftQty,ordata.price,ordata.isCanceled,ordata.userTag]
                writer.writerow(tmpdata)  # 写入数据
            
            
            
            
    def load_order_info(self):
        currrent_path = os.path.dirname(__file__)
        strTradingday=GetCurTradingDay()
        orderfile_path = os.path.join(currrent_path,strTradingday+'_order.csv')
        orderinfo=dict()
        
        #logger.info(orderfile_path)
        # 打开并读取csv文件
        
        if not os.path.isfile(orderfile_path):
            pass
        else:
            with open(orderfile_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    od=OrderData(row[0],row[1],bool(row[2]=='TRUE'),row[3],row[4],row[5],bool(row[6]=='TRUE'),row[7])
                    
                    '''
                    orderinfo[row[0]]={'localid':row[0],'code':row[1],
                                       'isBuy':row[2],'totalQty':row[3],
                                       'leftQty':row[4],'price':row[5],
                                       'isCanceled':row[6],'userTag':row[7]}
                    '''
                    orderinfo[int(row[0])]=od
            
        
        '''
        try:
            
            pass
                    
        except FileNotFoundError:
            print(f"文件不存在:{orderfile_path}")
        '''
        
    
        #print(orderinfo)
        return orderinfo

        
def GetCurTradingDay():
    tradingday=datetime.now()
    wd=datetime.now().weekday()
    h=datetime.now().hour
    m=datetime.now().minute
    if wd==5 or wd==6 or wd==0:
        if wd==5:
            if h*100+m>1530:
                tradingday=tradingday+timedelta(days=3)
        elif wd==6:
            tradingday=tradingday+timedelta(days=2)
        else:
            tradingday=tradingday+timedelta(days=1)
    else:
        if h*100+m>1530:
            tradingday=tradingday+timedelta(days=1)
        
    strTradingday=(f"{tradingday.year}{tradingday.month:02d}{tradingday.day:02d}")
    return strTradingday
        
        
       
        
        
        
        
        
        
        
        
        
        