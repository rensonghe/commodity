import rqdatac as rq
rq.init()
print('------------------------------'
      'begining loading data')
# for underlying_symbols_str in ('AU', 'SN', 'NI', 'SC','I', 'RB', 'ZN', 'HC', 'FU','OI','CU','AG','RU','BU'):
#     if underlying_symbols_str=='SN':
#         data = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str, start_date=20230801,
#                                               end_date=20230831, frequency='tick', fields=None, adjust_type='none',
#                                               adjust_method='prev_close_spread')
#         data.to_csv('/home/xianglake/SH_commodity_data/2308/data_{}_tick_2308.csv'.format(underlying_symbols_str),
#                      index=True, sep=',')
# #dce
# for underlying_symbols_str in ('I', 'FB', 'J', 'JM', 'M', 'P','PG','Y'):
#     # if underlying_symbols_str=='AG':
#     data = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str, start_date=20230701,
#                                           end_date=20230731, frequency='tick', fields=None, adjust_type='none',
#                                           adjust_method='prev_close_spread')
#     data.to_csv('/home/xianglake/DCE_commodity_data/2307/data_{}_tick_2307.csv'.format(underlying_symbols_str),
#                  index=True, sep=',')



# for underlying_symbols_str in ('I', 'FB', 'J', 'JM', 'M', 'P','PG','Y'):
#
#     for month_cal in range(1, 19):
#             # data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210101, end_date=20230701, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             # data2.to_csv('D:/BaiduNetdiskDownload/data_{}_tick_2021and23.csv'.format(underlying_symbols_str), index=True, sep=',')
#
#         if month_cal==1:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210101, end_date=20210131, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2101/data_{}_tick_2101.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==2:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210201, end_date=20210218, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2102/data_{}_tick_2102.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==3:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210301, end_date=20210331, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2103/data_{}_tick_2103.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==4:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210401, end_date=20210430, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2104/data_{}_tick_2104.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==5:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210501, end_date=20210531, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2105/data_{}_tick_2105.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==6:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210601, end_date=20210630, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2106/data_{}_tick_2106.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==7:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210701, end_date=20210731, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2107/data_{}_tick_2107.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==8:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210801, end_date=20210831, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2108/data_{}_tick_2108.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==9:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210901, end_date=20210930, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2109/data_{}_tick_2109.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==10:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211001, end_date=20211031, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2110/data_{}_tick_2110.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==11:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211101, end_date=20211130, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2111/data_{}_tick_2111.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==12:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211201, end_date=20211231, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2112/data_{}_tick_2112.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==13:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230101, end_date=20230131, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2301/data_{}_tick_2301.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==14:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230201, end_date=20230218, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2302/data_{}_tick_2302.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==15:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230301, end_date=20230331, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2303/data_{}_tick_2303.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==16:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230401, end_date=20230430, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2304/data_{}_tick_2304.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==17:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230501, end_date=20230531, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2305/data_{}_tick_2305.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==18:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230601, end_date=20230630, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2306/data_{}_tick_2306.csv'.format(underlying_symbols_str), index=True, sep=',')
#         if month_cal==19:
#             data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230701, end_date=20230731, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
#             data2.to_csv('/home/xianglake/DCE_commodity_data/2307/data_{}_tick_2307.csv'.format(underlying_symbols_str), index=True, sep=',')


for underlying_symbols_str in ('RB','NI','AU', 'SN', 'SC', 'ZN', 'HC', 'FU','CU','AG','BU'):
# for underlying_symbols_str in ('RB','NI'):

    for month_cal in range(8, 10):
            # data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210101, end_date=20230701, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
            # data2.to_csv('D:/BaiduNetdiskDownload/data_{}_tick_2021and23.csv'.format(underlying_symbols_str), index=True, sep=',')

        # if month_cal==1:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210101, end_date=20210131, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2101/data_{}_tick_2101.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==2:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210201, end_date=20210228, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2102/data_{}_tick_2102.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==3:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210301, end_date=20210331, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2103/data_{}_tick_2103.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==4:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210401, end_date=20210430, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2104/data_{}_tick_2104.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==5:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210501, end_date=20210531, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2105/data_{}_tick_2105.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==6:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210601, end_date=20210630, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2106/data_{}_tick_2106.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==7:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210701, end_date=20210731, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2107/data_{}_tick_2107.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==8:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210801, end_date=20210831, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2108/data_{}_tick_2108.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==9:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20210901, end_date=20210930, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2109/data_{}_tick_2109.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==10:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211001, end_date=20211031, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2110/data_{}_tick_2110.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==11:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211101, end_date=20211130, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2111/data_{}_tick_2111.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==12:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20211201, end_date=20211231, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2112/data_{}_tick_2112.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==13:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230101, end_date=20230131, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2301/data_{}_tick_2301.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==14:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230201, end_date=20230218, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2302/data_{}_tick_2302.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==15:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230301, end_date=20230331, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2303/data_{}_tick_2303.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==16:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230401, end_date=20230430, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2304/data_{}_tick_2304.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==17:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230501, end_date=20230531, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2305/data_{}_tick_2305.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==18:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230601, end_date=20230630, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2306/data_{}_tick_2306.csv'.format(underlying_symbols_str), index=True, sep=',')
        # if month_cal==19:
        #     data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230701, end_date=20230731, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
        #     data2.to_csv('/home/xianglake/SH_commodity_data/2307/data_{}_tick_2307.csv'.format(underlying_symbols_str), index=True, sep=',')
        if month_cal==8:
            data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230801, end_date=20230831, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
            data2.to_csv('/home/xianglake/SH_commodity_data/2308/data_{}_tick_2308.csv'.format(underlying_symbols_str), index=True, sep=',')
        if month_cal==9:
            data2 = rq.futures.get_dominant_price(underlying_symbols=underlying_symbols_str,start_date=20230901, end_date=20230930, frequency='tick',fields=None, adjust_type='none',adjust_method='prev_close_spread')
            data2.to_csv('/home/xianglake/SH_commodity_data/2309/data_{}_tick_2309.csv'.format(underlying_symbols_str), index=True, sep=',')



print('------------------------------'
      'finishing loading data')