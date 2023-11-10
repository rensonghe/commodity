# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:27:04 2023

@author: linyiSky
"""



class OrderData:
    def __init__(self,localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        '''交易参数'''
        
        self.localid = localid            
        self.code=stdCode
        self.isBuy=isBuy
        
        self.totalQty=totalQty
        self.leftQty=leftQty
        self.price=price
        self.isCanceled=isCanceled
        self.userTag=userTag        
        
        
        
class TradeData:
    def __init__(self,localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        '''交易参数'''
        
        self.localid = localid            
        self.code=stdCode
        self.isBuy=isBuy
        self.volume=qty
        self.price=price
        self.userTag=userTag        
        
        

class TickData:
    def __init__(self,newTick:dict):
        '''交易参数'''
        
        self.closetime = newTick['time']            #时间戳13位
        self.exchg=newTick['exchg']                 #交易所代码
        self.code=newTick['code']                   #合约代码
        
        self.price=newTick['price']                 #最新价
        self.open=newTick['open']
        self.high=newTick['high']
        self.low=newTick['low']
        self.settle_price=newTick['settle_price']
        
        self.upper_limit=newTick['upper_limit']     #涨停价
        self.lower_limit=newTick['lower_limit']     #跌停价
        
        self.total_volume=newTick['total_volume']   #交易日总成交量
        self.volume=newTick['volume']               #当前成交量
        
        self.total_turnover=newTick['total_turnover']   #总成交额
        self.diff_interest=newTick['diff_interest']
        
        self.open_interest=newTick['open_interest']     #持仓量
        self.turn_over=newTick['turn_over']             #本次成交额
        
        self.ask_price_1 = newTick['ask_price_0']  #askprice
        self.ask_volume_1 = newTick['ask_qty_0']  #askvol
        self.bid_price_1=newTick['bid_price_0']  #bidprice
        self.bid_volume_1=newTick['bid_qty_0']  #bidvol
        
        self.ask_price_2 = newTick['ask_price_1']  #askprice
        self.ask_volume_2 = newTick['ask_qty_1']  #askvol
        self.bid_price_2=newTick['bid_price_1']  #bidprice
        self.bid_volume_2=newTick['bid_qty_1']  #bidvol
        
        self.ask_price_3 = newTick['ask_price_2']  #askprice
        self.ask_volume_3 = newTick['ask_qty_2']  #askvol
        self.bid_price_3=newTick['bid_price_2']  #bidprice
        self.bid_volume_3=newTick['bid_qty_2']  #bidvol
        
        self.ask_price_4 = newTick['ask_price_3']  #askprice
        self.ask_volume_4 = newTick['ask_qty_3']  #askvol
        self.bid_price_4=newTick['bid_price_3']  #bidprice
        self.bid_volume_4=newTick['bid_qty_3']  #bidvol
        
        self.ask_price_5 = newTick['ask_price_4']  #askprice
        self.ask_volume_5 = newTick['ask_qty_4']  #askvol
        self.bid_price_5=newTick['bid_price_4']  #bidprice
        self.bid_volume_5=newTick['bid_qty_4']  #bidvol
        
        self.ask_price_6 = newTick['ask_price_5']  #askprice
        self.ask_volume_6 = newTick['ask_qty_5']  #askvol
        self.bid_price_6=newTick['bid_price_5']  #bidprice
        self.bid_volume_6=newTick['bid_qty_5']  #bidvol
        
        self.ask_price_7 = newTick['ask_price_6']  #askprice
        self.ask_volume_7 = newTick['ask_qty_6']  #askvol
        self.bid_price_7=newTick['bid_price_6']  #bidprice
        self.bid_volume_7=newTick['bid_qty_6']  #bidvol
        
        self.ask_price_8 = newTick['ask_price_7']  #askprice
        self.ask_volume_8 = newTick['ask_qty_7']  #askvol
        self.bid_price_8=newTick['bid_price_7']  #bidprice
        self.bid_volume_8=newTick['bid_qty_7']  #bidvol
        
        self.ask_price_9 = newTick['ask_price_8']  #askprice
        self.ask_volume_9 = newTick['ask_qty_8']  #askvol
        self.bid_price_9 =  newTick['bid_price_8']  #bidprice
        self.bid_volume_9 = newTick['bid_qty_8']  #bidvol
        
        self.ask_price_10 = newTick['ask_price_9']  #askprice
        self.ask_volume_10 = newTick['ask_qty_9']  #askvol
        self.bid_price_10=newTick['bid_price_9']  #bidprice
        self.bid_volume_10=newTick['bid_qty_9']  #bidvol
        
        
        
class BarData:
    def __init__(self,period:str,newBar:dict):
        '''交易参数'''
        
        self.period=period                         #周期
        self.date = newBar['date']            #
        self.reserve = newBar['reserve']
        self.open=newBar['open']
        self.high=newBar['high']
        self.low=newBar['low']
        self.close=newBar['close']

        self.settle=newBar['settle']   #
        self.volume=newBar['vol']               #当前成交量
        self.money=newBar['money']
        self.hold=newBar['hold']
        self.diff=newBar['diff']
        

        