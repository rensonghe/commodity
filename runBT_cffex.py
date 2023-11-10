import os
from wtpy import WtBtEngine, EngineType
from strategies.HftStraDemo import HftStraDemo
from wtpy.apps import WtBtAnalyst
from strategies.Ht_ai_strategy_demo_bt_cffex import Ht_ai_strategy_demo_bt
from store_output_files import clean_history_file, rename_files, retain_files
#from strategies.LogUtils import LogUtils



if __name__ == "__main__":
    # 创建一个运行环境，并加入策略
    engine = WtBtEngine(EngineType.ET_HFT)
    engine.init('../common/', "configbt.yaml")
    engine.configBacktest(202304030900, 202306030000)
    engine.configBTStorage(mode="csv", path="../storage/")
    engine.commitBTConfig()

    name = 'if'
    exg = 'CFFEX'
    bar = '25'
    out_threshold = '50'
    x = '0.001'
    dollar = '10000000'
    signal_file = '/home/xianglake/songhe/future_backtest/{}300/CFFEX_{}300_20230403_0603_{}bar_vwapv_5s_ST1.0_20231017_filter_90_{}_{}_{}.csv'\
        .format(name.upper(),name.upper(),bar, out_threshold, x, dollar)
    
    code = exg + '.' + name

    straInfo = Ht_ai_strategy_demo_bt(name='bc', code='INE.bc', expsecs=5, offset=0, signal_file=signal_file, freq=10)
    engine.set_hft_strategy(straInfo)

    clean_history_file("./outputs_bt/{}/".format(name))

    engine.run_backtest(bAsync=True)

    # # 创建绩效分析模块
    # analyst = WtBtAnalyst()
    # # 将回测的输出数据目录传递给绩效分析模块
    # analyst.add_strategy("iron", folder="./outputs_bt/", init_capital=1000000, rf=0.02, annual_trading_days=240)
    # # 运行绩效模块
    # analyst.run_new()

    kw = input('press any key to exit\n')
    engine.release_backtest()

    # rename_output_files
    rename_files("./outputs_bt/{}/".format(name), '.', '_{}.'.format(code))

    # copy_to_history
    retain_files("./outputs_bt/", name)
