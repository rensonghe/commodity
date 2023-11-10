#%%
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import datetime

from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
#%%
future = 'I'
exchange = 'dce'
dollar = 3_000_000
data = pd.read_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor_%s.csv'%(exchange, future, future, dollar))

data = data.sort_values(by='datetime', ascending=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data['time'] = data['datetime'].dt.strftime('%H:%M:%S')
def time_interval(data):
    start_time = datetime.datetime.strptime('09:05:00', '%H:%M:%S').time()
    end_time = datetime.datetime.strptime('10:15:00','%H:%M:%S').time()

    start_time2 = datetime.datetime.strptime('13:30:00', '%H:%M:%S').time()
    end_time2 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()

    start_time3 = datetime.datetime.strptime('21:05:00', '%H:%M:%S').time()
    end_time3 = datetime.datetime.strptime('22:59:59','%H:%M:%S').time()

    start_time1 = datetime.datetime.strptime('10:30:00', '%H:%M:%S').time()
    end_time1 = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

    start_time4 = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
    end_time4 = datetime.datetime.strptime('00:59:59','%H:%M:%S').time()

    data_time = data[(data.datetime.dt.time >= start_time) & (data.datetime.dt.time <= end_time)|
                     (data.datetime.dt.time >= start_time1) & (data.datetime.dt.time <= end_time1)|
                     (data.datetime.dt.time >= start_time2) & (data.datetime.dt.time <= end_time2)|
                    (data.datetime.dt.time >= start_time3) & (data.datetime.dt.time <= end_time3)|
                     (data.datetime.dt.time >= start_time4) & (data.datetime.dt.time <= end_time4)]

    data_time = data_time.sort_values(by='datetime', ascending=True)
    return data_time
data = time_interval(data)
data = data.set_index('datetime')
#%%
from ta.volume import *
from ta.volatility import *
from ta.trend import *
from ta.momentum import *
def indicator_function(df):
    df['indicator_FI'] = force_index(close=df['price'], volume=df['size'], window=13)
    df['indicator_EoM'] = ease_of_movement(high=df['high'], low=df['low'], volume=df['size'], window=14)
    df['indicator_VPT'] = volume_price_trend(close=df['price'], volume=df['size'])
    df['indicator_BBH'] = bollinger_hband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBL'] = bollinger_lband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBM'] = bollinger_mavg(close=df['price'], window=20)
    df['indicator_BBP'] = bollinger_pband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBW'] = bollinger_wband(close=df['price'], window=20, window_dev=2)
    df['indicator_DCH'] = donchian_channel_hband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCL'] = donchian_channel_lband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCM'] = donchian_channel_mband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCW'] = donchian_channel_wband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCP'] = donchian_channel_pband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_KCL'] = keltner_channel_lband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCH'] = keltner_channel_hband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCW'] = keltner_channel_wband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCP'] = keltner_channel_pband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCM'] = keltner_channel_mband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_MACD'] = macd(close=df['price'], window_fast=26, window_slow=12)
    df['indicator_MACD_diff'] = macd_diff(close=df['price'], window_fast=26, window_slow=12, window_sign=9)
    df['indicator_MACD_signal'] = macd_signal(close=df['price'], window_fast=26, window_slow=12, window_sign=9)
    df['indicator_SMA_fast'] = sma_indicator(close=df['price'], window=16)
    df['indicator_SMA_slow'] = sma_indicator(close=df['price'], window=32)
    df['indicator_RSI'] = stochrsi(close=df['price'], window=9, smooth1=16, smooth2=26)
    df['indicator_RSIK'] = stochrsi_k(close=df['price'], window=9, smooth1=16, smooth2=26)
    df['indicator_RSID'] = stochrsi_d(close=df['price'], window=9, smooth1=16, smooth2=26)

    return df
data = data.rename({'highest':'high', 'lowest':'low'}, axis='columns')
data = indicator_function(data)
#%%
bar = 15
x = 0.0005
data['target'] = np.log(data['mid_price']/data['mid_price'].shift(bar))
data['target'] = data['target'].shift(-bar)

def classify(y):

    if y < -x:
        return 0
    if y > x:
        return 1
    else:
        return -1
print(data['target'].apply(lambda x:classify(x)).value_counts())
print(len(data[data['target'].apply(lambda x:classify(x))==-1])/len(data['target'].apply(lambda x:classify(x))))
#%%
def calcpearsonr(data,rolling):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[7:88]):

        ic = data[column].rolling(rolling).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)
        IC = pd.DataFrame(ic_list)
        columns = pd.DataFrame(data.columns[7:88])
        IC_columns = pd.concat([IC, columns], axis=1)
        col = ['value', 'factor']
        IC_columns.columns = col
    return IC_columns
IC_columns = calcpearsonr(data,rolling=bar)
#%%
time_1 = '2023-05-01 09:00:00'
time_2 = '2023-05-30 01:00:00'

cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[data.index < time_1]
test = data[(data.index >= time_1)&(data.index <= time_2)]
train['target'] = train['target'].apply(lambda x:classify(x))
train = train[~train['target'].isin([-1])]
train_set = train[train_col]
train_set = train_set.iloc[:,7:89] #65
train_target = train["target"]
test_set = test[train_col]
test_set = test_set.iloc[:,7:89]
test_target = test["target"]
print(test['target'].apply(lambda x:classify(x)).value_counts())
print(len(test[test['target'].apply(lambda x:classify(x))==-1])/len(test['target'].apply(lambda x:classify(x))))

test_ = test.copy()
test_['target'] = test_['target'].apply(lambda x:classify(x))
test_ = test_[~test_['target'].isin([-1])]
test_set_ = test_[train_col]
test_set_ = test_set_.iloc[:,7:89]
test_target_ = test_["target"]

X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)
X_test_ = np.array(test_set_)
X_test_target_ = np.array(test_target_)

del train_set, test_set, train_target, test_target, test_set_, test_target_, test_
df = test
df['min'] = ((df['closetime']-df['closetime'].shift(bar))/1000)/60
print(df['min'].describe())
del df
#%%
def custom_smooth_l1_loss_eval(y_pred, lgb_train):
    """
    Calculate loss value of the custom loss function
     Args:
        y_true : array-like of shape = [n_samples] The target values.
        y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        loss: loss value
        is_higher_better : bool, loss是越低越好，所以这个返回为False
        Is eval result higher better, e.g. AUC is ``is_higher_better``.
    """
    y_true = lgb_train.get_label()
    # y_pred = y_pred.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    loss = np.where(np.abs(residual) < 1, (residual ** 2) * 0.5, np.abs(residual) - 0.5)
    return "custom_asymmetric_eval", np.mean(loss), False

def custom_smooth_l1_loss_train(y_pred, lgb_train):
    """Calculate smooth_l1_loss
    Args:
        y_true : array-like of shape = [n_samples]
        The target values. y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        grad: gradient, should be list, numpy 1-D array or pandas Series
        hess: matrix hessian value
    """
    y_true = lgb_train.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    grad = np.where(np.abs(residual) < 1, residual, 1)
    hess = np.where(np.abs(residual) < 1, 1.0, 0.0)
    return grad, hess
#%% first model
def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', pearsonr(preds, train_data.get_label())[0], is_higher_better
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample):
    # LightGBM expects next three parameters need to be integer.
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    learning_rate = float(learning_rate)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    n_estimators = int(n_estimators)
    min_child_samples = float(min_child_samples)
    min_split_gain = float(min_split_gain)
    # scale_pos_weight = float(scale_pos_weight)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    kf = TimeSeriesSplit(n_splits=5)
    X_train_pred = np.zeros(len(X_train_target))


    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 2 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)
        params = {
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'min_split_gain': min_split_gain,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'subsample': subsample,
            'n_estimators': n_estimators,
            # 'learning_rate' : learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'cross_entropy',
            # 'objective': 'multiclass',
            # 'num_class': '3',
            'save_binary': True,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'boosting_type': 'gbdt',
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            'metric': {'cross_entropy','auc'},
            # 'metric': {'multi_logloss','auc'},
            'num_threads': 25}


        model = lgb.train(params, train_set=train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100) #fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)
        X_train_pred += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        # fpr_train, tpr_train, thresholds_train = roc_auc_score(x_val, y_val)
        # gmeans_train = sqrt(tpr_train * (1 - fpr_train))
        # ix_train = argmax(gmeans_train)
        # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
        #
        # thresholds_point_train = thresholds_train[ix_train]
        # x_val_thresholds = [1 if y > thresholds_point_train else 0 for y in x_val]
        score = roc_auc_score(X_train_target, X_train_pred)

        # score = bayesian_ic_lgbm(X_train_pred, X_train_target)

        return score

bounds_LGB = {
    'colsample_bytree': (0.7, 1),
    'n_estimators': (500, 10000),
    'num_leaves': (31, 500),
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.001, 0.3),
    'min_child_weight': (0.00001, 0.01),
    'min_child_samples': (2, 100),
    'min_split_gain': (0.1, 1),
    'subsample': (0.7, 1),
    'reg_alpha': (1, 2),
    'reg_lambda': (1, 2),
    'max_depth': (-1, 50),
    # 'scale_pos_weight':(0.5, 10)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

# LGB_BO.max['target']
# LGB_BO.max['params']

# first model
def lightgbm_model(X_train, X_train_target, X_test, X_test_target, X_test_, X_test_target_, LGB_BO, train):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_ = np.zeros(len(X_test_target_))
    y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []
    LGB_BO.max['params'] = LGB_BO.max['params']
    features = train.iloc[:, 7:89].columns
    features = list(features)

    def plot_importance(importances, features, PLOT_TOP_N=20, figsize=(20, 20)):
        importance_df = pd.DataFrame(data=importances, columns=features)
        sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
        sorted_importance_df = importance_df.loc[:, sorted_indices]
        plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
        _, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_xscale('log')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        sns.boxplot(data=sorted_importance_df[plot_cols],
                    orient='h',
                    ax=ax)
        plt.show()

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)


        params = {
            'boosting_type': 'gbdt',
            # 'metric': 'multi_logloss',
            # 'objective': 'multiclass',
            'metric': {'cross_entropy','auc','average_precision',},
            'objective': 'binary',  # regression,binary,multiclass
            # 'num_class': 3,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'num_leaves': int(LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(LGB_BO.max['params']['learning_rate']),
            'max_depth': int(LGB_BO.max['params']['max_depth']),
            'n_estimators': int(LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(LGB_BO.max['params']['subsample']),
            'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 30
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        y_pred_ += model.predict(X_test_, num_iteration=model.best_iteration) / kf.n_splits
        y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list, y_pred_

y_pred, y_pred_train, model_list,y_pred_= lightgbm_model(X_train=X_train, X_train_target=X_train_target, X_test=X_test, X_test_target=X_test_target,
                                                  X_test_=X_test_, X_test_target_=X_test_target_,LGB_BO=LGB_BO, train=train)
#%%
def first_model_train_test(X_train_target, first_y_pred_train, X_test_target_, first_y_pred):

    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    from numpy import sqrt, argmax
    fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, first_y_pred_train)
    gmeans_train = sqrt(tpr_train * (1-fpr_train))
    ix_train = argmax(gmeans_train)
    # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
    thresholds_point_train = thresholds_train[ix_train]
    first_yhat_train = [1 if y > thresholds_point_train else 0 for y in first_y_pred_train]
    # print("secondary_model训练集表现：")
    # print(classification_report(yhat_train,X_train_target))

    fpr, tpr, thresholds = roc_curve(X_test_target_, first_y_pred)
    # fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # thresholds_point = thresholds_train[ix_train]
    first_yhat = [1 if y > thresholds[ix] else 0 for y in first_y_pred]
    # yhat = [1 if y > 0.55 else 0 for y in y_pred]
    # print("secondary_model测试集表现：")
    # print(classification_report(secondary_yhat,X_test_target))
    # print(metrics.confusion_matrix(yhat, X_test_target))
    # print('AUC:', metrics.roc_auc_score(secondary_yhat, X_test_target))
    return first_yhat_train, first_yhat
first_yhat_train, first_yhat = first_model_train_test(X_train_target, y_pred_train, X_test_target_, y_pred_)
print("first_model训练集表现：")
print(classification_report(first_yhat_train,X_train_target))
print("first_model测试集表现：")
print(classification_report(first_yhat,X_test_target_))
print('AUC:', metrics.roc_auc_score(first_yhat,X_test_target_))
#%%
test_data = test.copy()
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred,columns=['predict'])
predict['closetime'] = test_data['closetime']
# predict['vwapv'] = test_data['vwapv_15s']
predict['price'] = test_data['price']
predict['target'] = test_data['target']
# predict['pctrank'] = predict.index.map(lambda x : predict.loc[:x].predict.rank(pct=True)[x])
def pctrank(x):
    n = len(x)
    temp = x.argsort()
    ranks = np.empty(n)
    ranks[temp] = (np.arange(n) + 1) / n
    return ranks[-1]
# predict['pctrank'] = predict['predict'].rolling(bar).apply(pctrank)
# df_1 = predict.loc[predict['pctrank']>0.9]
# df_0 = predict.loc[predict['pctrank']<0.1]
# print(len(df_1))
# print(len(df_0))
#
df_1 = predict.loc[predict['predict']>np.percentile(y_pred_train[-20000:], 95)]
df_0 = predict.loc[predict['predict']<np.percentile(y_pred_train[-20000:], 5)]
print(len(df_1))
print(len(df_0))
df_1['side'] = 'buy'
df_0['side'] = 'sell'
final_df = pd.concat([df_1, df_0], axis=0)
final_df = final_df.sort_values(by='closetime', ascending=True)
final_df = final_df.reset_index(drop=True)
print(final_df.loc[:,['target','predict']].corr())
# print(pearsonr(final_df['target'],final_df['predict'])[0])
#%%
signal = test.reset_index()
signal['predict'] = predict['predict']
signal_1 = signal[signal['predict']>=np.percentile(y_pred_train[-20000:], 95)]
signal_0 = signal[signal['predict']<=np.percentile(y_pred_train[-20000:], 5)]
# signal['pctrank'] = predict['pctrank']
# signal_1 = signal[signal['pctrank']>0.9]
# signal_0 = signal[signal['pctrank']<0.1]
signal_1['side'] = 'buy'
signal_0['side'] = 'sell'
signal_df = pd.concat([signal_1, signal_0],axis=0)
signal_df = signal_df.sort_values(by='closetime', ascending=True)
signal_df = signal_df.set_index('datetime')

def abs_classify(y):

    if y > x:
        return 1
    else:
        return 0

train = data[data.index < time_1]
train_set = train[train_col]
train_set = train_set.iloc[:,7:89]
# train_set = train_set.iloc[:,7:65]
train_target = abs(train['target']).apply(lambda x:abs_classify(x))
test_set = signal_df[train_col]
test_set = test_set.iloc[:,7:89]#65
test_target = abs(signal_df["target"]).apply(lambda x:abs_classify(x))
print(signal_df['target'].apply(lambda x:abs_classify(x)).value_counts())
print(len(signal_df[signal_df['target'].apply(lambda x:abs_classify(x))==1])/len(signal_df['target'].apply(lambda x:abs_classify(x))))
#%%
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)

#
secondary_LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    secondary_LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

def secondary_lightgbm_model(X_train, X_train_target, X_test, X_test_target, LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []
    LGB_BO.max['params'] = secondary_LGB_BO.max['params']

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)


        params = {
            'boosting_type': 'gbdt',
            # 'metric': 'multi_logloss',
            # 'objective': 'multiclass',
            'metric': {'cross_entropy','auc','average_precision',},
            'objective': 'binary',  # regression,binary,multiclass
            # 'num_class': 3,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'num_leaves': int(LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(LGB_BO.max['params']['learning_rate']),
            'max_depth': int(LGB_BO.max['params']['max_depth']),
            'n_estimators': int(LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(LGB_BO.max['params']['subsample']),
            'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 28
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list

secondary_y_pred, secondary_y_pred_train, secondary_model_list = secondary_lightgbm_model(X_train=X_train, X_train_target=X_train_target,
                                                                                          X_test=X_test, X_test_target=X_test_target, LGB_BO=secondary_LGB_BO)
 #%% two model saving
def model_saveing(model_list,secondary_model_list,base_path, future):

    joblib.dump(model_list[0],'{}/{}_lightGBM_side_0.pkl'.format(base_path, future))
    joblib.dump(model_list[1],'{}/{}_lightGBM_side_1.pkl'.format(base_path,future))
    joblib.dump(model_list[2],'{}/{}_lightGBM_side_2.pkl'.format(base_path,future))
    joblib.dump(model_list[3],'{}/{}_lightGBM_side_3.pkl'.format(base_path,future))
    joblib.dump(model_list[4],'{}/{}_lightGBM_side_4.pkl'.format(base_path,future))
    joblib.dump(secondary_model_list[0],'{}/{}_lightGBM_out_0.pkl'.format(base_path,future))
    joblib.dump(secondary_model_list[1],'{}/{}_lightGBM_out_1.pkl'.format(base_path,future))
    joblib.dump(secondary_model_list[2],'{}/{}_lightGBM_out_2.pkl'.format(base_path,future))
    joblib.dump(secondary_model_list[3],'{}/{}_lightGBM_out_3.pkl'.format(base_path,future))
    joblib.dump(secondary_model_list[4],'{}/{}_lightGBM_out_4.pkl'.format(base_path,future))

    return
base_path = '/home/xianglake/songhe/future_saved_model/'
model_saveing(model_list, secondary_model_list, base_path, future)
#%%
def seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred):

    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    from numpy import sqrt, argmax
    fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, secondary_y_pred_train)
    gmeans_train = sqrt(tpr_train * (1-fpr_train))
    ix_train = argmax(gmeans_train)
    # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
    thresholds_point_train = thresholds_train[ix_train]
    secondary_yhat_train = [1 if y > thresholds_point_train else 0 for y in secondary_y_pred_train]
    # print("secondary_model训练集表现：")
    # print(classification_report(yhat_train,X_train_target))

    fpr, tpr, thresholds = roc_curve(X_test_target, secondary_y_pred)
    # fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # thresholds_point = thresholds_train[ix_train]
    secondary_yhat = [1 if y > thresholds[ix] else 0 for y in secondary_y_pred]
    # yhat = [1 if y > 0.55 else 0 for y in y_pred]
    # print("secondary_model测试集表现：")
    # print(classification_report(secondary_yhat,X_test_target))
    # print(metrics.confusion_matrix(yhat, X_test_target))
    # print('AUC:', metrics.roc_auc_score(secondary_yhat, X_test_target))
    return secondary_yhat_train, secondary_yhat
secondary_yhat_train, secondary_yhat = seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred)
print("secondary_model训练集表现：")
print(classification_report(secondary_yhat_train,X_train_target))
print("secondary_model测试集表现：")
print(classification_report(secondary_yhat,X_test_target))
print('AUC:', metrics.roc_auc_score(secondary_yhat,X_test_target))
#%%
out_threshold = 70
secondary_predict = pd.DataFrame(secondary_y_pred, columns=['out'])
# secondary_predict['out_pctrank'] = secondary_predict.index.map(lambda x : secondary_predict.loc[:x].out.rank(pct=True)[x])
signal_df_ = signal_df.reset_index()
signal_df_['out'] = secondary_predict['out']
# signal_df_['out_pctrank'] = secondary_predict['out_pctrank']
final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train[-20000:], out_threshold)]
final_df['acc'] = np.where(abs(final_df['target'])>x,1,0)
print((len(final_df[final_df['acc']==1])/len(final_df))*100)
print('信号长度:',len(final_df))
final_df = final_df.sort_values(by='closetime', ascending=True)
final_df = final_df.loc[:,['datetime','closetime','mid_price','price','predict','target','side']]
print(final_df.loc[:,['target','predict']].corr())
# print(pearsonr(final_df['target'],final_df['predict'])[0])
#%%
# final_df = final_df.dropna(axis=0)
# final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
# final_df['time'] = None
# from datetime import datetime
# for i in range(len(final_df['datetime'])):
#     final_df['time'].iloc[i] = ((final_df['datetime'].iloc[i].replace(second=59, microsecond=999999).timestamp()-28800)*1000)
#     if final_df['time'].iloc[i] % 100 == 0:
#         final_df['time'].iloc[i] = (final_df['time'].iloc[i])-1
# final_df['closetime'] = final_df['time'].astype('int')
# del final_df['time']
final_df = final_df.set_index('datetime')
final_df['closetime'] = final_df['closetime'].astype('int64')
final_df.to_csv('/home/xianglake/songhe/future_backtest/{}/DCE_{}_20230501_0530_{}bar_midprice_ST2.0_20230731_filter_{}.csv'.format(future, future, bar, out_threshold))
#%%
out_threshold = 80
df1 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20221212_0203_{}bar_vwap30s_ST1.0_20230709_filter_{}.csv'.format(future, future, bar, out_threshold))
df2 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230206_0331_{}bar_vwap30s_ST1.0_20230709_filter_{}.csv'.format(future, future, bar, out_threshold))
df3 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230403_0603_{}bar_vwap30s_ST1.0_20230709_filter_{}.csv'.format(future, future, bar, out_threshold))
final_df = pd.concat([df1,df2,df3],axis=0)
del df1,df2,df3
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
final_df.to_csv('/root/songhe/future_backtest/{}/SHFE_{}_20221212_0603_{}bar_vwap30s_ST1.0_20230709_filter_{}.csv'.format(future, future, bar, out_threshold))
#%%
df1 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230104_0203_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df2 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230206_0304_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df3 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230306_0331_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df4 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230403_0428_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
df5 = pd.read_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230501_0603_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
final_df = pd.concat([df1,df2,df3,df4,df5],axis=0)
del df1,df2,df3,df4,df5
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
final_df.to_csv('/root/songhe/future_backtest/{}/SHFE_{}_20230104_0603_{}bar_vwap15s_ST1.0_20230701_filter_{}.csv'.format(future, future, bar, out_threshold))
#%%
final_df = pd.read_csv('/songhe/future/AI/SHFE_{}_20220502_0527_{}bar_vwap_ST2.0_20230530_filter_{}.csv'.format(future, bar, out_threshold))
final_df = final_df.iloc[:,1:]
final_df = final_df.rename({'closetime':'timestamp'},axis='columns')
final_df['symbol'] = future

from pyarrow import Table

minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
dir_name = 'datafile/eval/songhe/{}'.format("cta_future_SHFE_{}_{}bar_vwap_20220502_0527_20230530_{}".format(future, bar, out_threshold))
pq.write_to_dataset(Table.from_pandas(df=final_df),
                    root_path=dir_name,
                    filesystem=minio, basename_template="part-{i}.parquet",
                    existing_data_behavior="overwrite_or_ignore")