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
import joblib
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
#%%
future = 'AG'
exchange = 'shfe'
dollar = 12_000_000
min = 40

# data = pd.read_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor_%s_2.csv'%(exchange, future, future, dollar))
data_2021 = pd.read_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor_%s_%smin_21.csv'%(exchange, future, future, dollar, min))
data_2022 = pd.read_csv('/home/xianglake/songhe/%s/%s/tick_factor/%s_tick_factor_%s_%smin_22.csv'%(exchange, future, future, dollar, min))
data = pd.concat([data_2021, data_2022],axis=0)
del data['datetime']
del data_2022, data_2021
# data = data_2022.copy()
data['datetime'] = pd.to_datetime(data['closetime']+28800000, unit='ms')
data['time'] = data['datetime'].dt.strftime('%H:%M:%S')
#
def time_interval(data):
    start_time = datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()
    end_time = datetime.datetime.strptime('10:15:00','%H:%M:%S').time()

    start_time2 = datetime.datetime.strptime('13:30:00', '%H:%M:%S').time()
    end_time2 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()

    start_time3 = datetime.datetime.strptime('21:00:00', '%H:%M:%S').time()
    end_time3 = datetime.datetime.strptime('23:59:59','%H:%M:%S').time()

    start_time1 = datetime.datetime.strptime('10:30:00', '%H:%M:%S').time()
    end_time1 = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

    start_time4 = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
    end_time4 = datetime.datetime.strptime('02:29:59','%H:%M:%S').time()

    data_time = data[(data.datetime.dt.time >= start_time) & (data.datetime.dt.time <= end_time)|
                     (data.datetime.dt.time >= start_time1) & (data.datetime.dt.time <= end_time1)|
                     (data.datetime.dt.time >= start_time2) & (data.datetime.dt.time <= end_time2)|
                    (data.datetime.dt.time >= start_time3) & (data.datetime.dt.time <= end_time3)|
                     (data.datetime.dt.time >= start_time4) & (data.datetime.dt.time <= end_time4)]

    data_time = data_time.sort_values(by='datetime', ascending=True)
    return data_time
data = time_interval(data)
# data = data.set_index('datetime')
data = data.sort_values(by='closetime', ascending=True)
data = data.set_index('datetime')

#%%
vwapv = 'vwapv_60s'
data['target'] = np.where((data[vwapv]<=data['high_2min'])&(data[vwapv]>=data['low_2min']),1,0)
print(len(data[data['target']==1])/len(data['target']))
#%%
time_1 = '2023-04-03 09:00:00'
time_2 = '2023-06-03 03:00:00'

cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[data.index < time_1]
test = data[(data.index >= time_1)&(data.index <= time_2)]

train_set = train[train_col]
train_set = train_set.iloc[:,5:115] #65
train_target = train["target"]

test_set = test[train_col]
test_set = test_set.iloc[:,5:115]
test_target = test["target"]
print(test['target'].value_counts())
print(len(test[test['target']==1])/len(test['target']))

X_train = np.array(train_set)
X_train_target = np.array(train_target)

X_test = np.array(test_set)
X_test_target = np.array(test_target)

#
del train_set, test_set, train_target, test_target
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
# bounds_LGB = {k: tuple(v) for k, v in bounds_LGB.items()}
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
def lightgbm_model(X_train, X_train_target, X_test, X_test_target, LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))

    y_pred_train = np.zeros(len(X_train_target))

    importances = []
    model_list = []
    LGB_BO.max['params'] = LGB_BO.max['params']
    features = train.iloc[:, 5:107].columns
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
        y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        # importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list,

y_pred, y_pred_train, model_list= lightgbm_model(X_train=X_train, X_train_target=X_train_target, X_test=X_test, X_test_target=X_test_target,LGB_BO=LGB_BO)

def first_model_train_test(X_train_target, first_y_pred_train, X_test_target, first_y_pred):

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

    fpr, tpr, thresholds = roc_curve(X_test_target, first_y_pred)
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
first_yhat_train, first_yhat = first_model_train_test(X_train_target, y_pred_train, X_test_target, y_pred)
print("first_model训练集表现：")
print(classification_report(first_yhat_train,X_train_target))
print("first_model测试集表现：")
print(classification_report(first_yhat,X_test_target))
print('AUC:', metrics.roc_auc_score(first_yhat,X_test_target))

#%%
predict = pd.DataFrame(y_pred, columns=['predict'])
signal = test.reset_index()
signal['predict'] = predict['predict']
singal_ = signal.copy()
singal_ = singal_.loc[:,['datetime','closetime',vwapv,'price','predict','target']]
#%%
out_threshold_set = [10,20,30,40,50,60]

for out_threshold in out_threshold_set:
    final_df_ = singal_[singal_['predict']>=np.percentile(y_pred_train[-300000:], out_threshold)]
    final_df_['success'] = 'success'
    print('--------------------'
          'out_threshold:',out_threshold)
    print('信号准确率:',(len(final_df_[final_df_['target']==1])/len(final_df_))*100)
    print('信号长度:',len(final_df_))
    final_df_ = final_df_.sort_values(by='closetime', ascending=True)
    final_df_ = final_df_.loc[:,['datetime','closetime',vwapv,'price','predict','target','success']]

    final_df = final_df_.set_index('datetime')
    final_df['closetime'] = final_df['closetime'].astype('int64')
    final_df.to_csv('/home/xianglake/songhe/future_backtest/{}/SHFE_{}_20230403_0603_{}_ST1.0_20230906_success_{}_{}.csv'
                    .format(future, future, vwapv,out_threshold, dollar))
#%%
import pandas as pd
import numpy as np
symbol = 'AG'
amount = '12000000'
bar = '15'
threshold = '0.0007'
out_threshold = '80'
success_threshold = '60'
# signal = pd.read_csv('/home/xianglake/songhe/future_backtest/AG/SHFE_AG_20230206_0331_30bar_vwapv_30s_ST1.1_20230831_filter_50_0.001_10000000.csv')
signal = pd.read_csv('/home/xianglake/songhe/future_backtest/{}/SHFE_{}_20230403_0603_{}bar_vwapv_60s_ST2.0_20230906_pctrank_{}_{}_{}.csv'
                     .format(symbol,symbol,bar,out_threshold, threshold, amount))
success = pd.read_csv('/home/xianglake/songhe/future_backtest/{}/SHFE_{}_20230403_0603_vwapv_60s_ST1.0_20230906_success_{}_{}.csv'
                      .format(symbol,symbol,success_threshold,amount))
merged_df = pd.merge(signal, success[['closetime', 'success']], on='closetime', how='left')
merged_df['success'] = np.where(merged_df['success']=='success',merged_df['success'], 'not_success')
merged_df = merged_df.set_index('datetime')
# merged_df.to_csv('/home/xianglake/songhe/future_backtest/AG/SHFE_AG_20230206_0331_30bar_vwapv_30s_ST1.1_20230831_filter_50_0.001_10000000_success.csv')
merged_df.to_csv('/home/xianglake/songhe/future_backtest/{}/SHFE_{}_20230403_0603_{}bar_vwapv_60s_ST2.1_20230906_pctrank_{}_{}_{}_success.csv'
                 .format(symbol, symbol, bar, out_threshold, threshold,amount))


