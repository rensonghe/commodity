import os
from wtpy import WtBtEngine, EngineType
from strategies.HftStraDemo import HftStraDemo
from wtpy.apps import WtBtAnalyst
from strategies.Ht_ai_strategy_demo_bt_dce import Ht_ai_strategy_demo_bt

#from strategies.LogUtils import LogUtils

def clean_history_file(folder_path):
    if os.path.exists(folder_path) is True:
        for filename in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, filename))  # 删除原文件


def rename_files(folder_path, old_ext, new_ext):
    # 获取目标文件夹下的所有文件名
    for filename in os.listdir(folder_path):
        new_filename = filename.replace(old_ext, new_ext)
        # 重命名文件
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        # os.remove(os.path.join(folder_path, filename))  # 删除原文件(正常来说会自动删除)

if __name__ == "__main__":
    # 创建一个运行环境，并加入策略
    engine = WtBtEngine(EngineType.ET_HFT)
    engine.init('../common/', "configbt.yaml")
    engine.configBacktest(202305010500, 202305310000)
    engine.configBTStorage(mode="csv", path="../storage/")
    engine.commitBTConfig()

    name = 'i'
    exg = 'DCE'
    signal_file = 'DCE_I_20230501_0530_15bar_midprice_ST2.0_20230731_filter_70.csv'
    code = exg + '.' + name

    straInfo = Ht_ai_strategy_demo_bt(name=name, code=code, expsecs=5, offset=0, signal_file=signal_file, freq=10)
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
