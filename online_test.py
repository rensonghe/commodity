
depth_dict = {'closetime': tick.closetime //100*100+99,
                           'ask_price1': tick.ask_price_1,'ask_size1': tick.ask_volume_1,'bid_price1': tick.bid_price_1,'bid_size1': tick.bid_volume_1,
                           'ask_price2': tick.ask_price_2,'ask_size2': tick.ask_volume_2,'bid_price2': tick.bid_price_2,'bid_size2': tick.bid_volume_2,
                           'ask_price3': tick.ask_price_3,'ask_size3': tick.ask_volume_3,'bid_price3': tick.bid_price_3,'bid_size3': tick.bid_volume_3,
                           'ask_price4': tick.ask_price_4,'ask_size4': tick.ask_volume_4,'bid_price4': tick.bid_price_4,'bid_size4': tick.bid_volume_4,
                           'ask_price5': tick.ask_price_5,'ask_size5': tick.ask_volume_5,'bid_price5': tick.bid_price_5,'bid_size5': tick.bid_volume_5,
                           'ask_price6': tick.ask_price_6,'ask_size6': tick.ask_volume_6,'bid_price6': tick.bid_price_6,'bid_size6': tick.bid_volume_6,
                           'ask_price7': tick.ask_price_7,'ask_size7': tick.ask_volume_7,'bid_price7': tick.bid_price_7,'bid_size7': tick.bid_volume_7,
                           'ask_price8': tick.ask_price_8,'ask_size8': tick.ask_volume_8,'bid_price8': tick.bid_price_8,'bid_size8': tick.bid_volume_8,
                           'ask_price9': tick.ask_price_9,'ask_size9': tick.ask_volume_9,'bid_price9': tick.bid_price_9,'bid_size9': tick.bid_volume_9,
                           'ask_price10': tick.ask_price_10,'ask_size10': tick.ask_volume_10,'bid_price10': tick.bid_price_10,'bid_size10': tick.bid_volume_10,
                            }
print(depth_dict)