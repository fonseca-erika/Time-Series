from models import Exponential_Smoothing, Arima, EnsembleModels, FBProphet, RandomForest, XGBoost, SupportVectorRegressor
import warnings

from dataload import DataLoad
from analytics import DataVisualization
from evaluation import EvaluateModel
from preprocessing import FeatureEngineering

warnings.simplefilter("ignore")
print('Time Series model composition')

# Loading Data
DL = DataLoad()
df = DL.load_csv(file_path='data/ross_train.csv', date='Date', target = 'Sales', show_details=True)

print(df)
# Analyzing using different frequencies
x = []
y = []
title = []
# Daily
x.append(df.index)
y.append(df.Sales)
title.append('Daily')
freq = {'W': 'Weekly',
        'M': 'Monthly',
        'Y': 'Yearly'}
for f in freq.keys():
    sampled_df = df.resample(f).sum()
    x.append(sampled_df.index)
    y.append(sampled_df.Sales)
    title.append(freq[f])

DataVisualization().multiple_plots(x, y, title, cols=2)

# ======================================================================================================================
# Exponential Smoothing
# ======================================================================================================================
model_es = Exponential_Smoothing(DL)
model_es.fit(trend_effect='add', damped=True, seasonal_effect='add', seasonality=7, use_boxcox=False, frequency='D')
forecast_es = model_es.predict(start=model_es.test.index[0], end=model_es.test.index[-1])

eval_es = EvaluateModel(model_es.test[model_es.target], forecast_es)
print('RMSE for exponential smoothing model: ', eval_es.rmse())
eval_es.plot(label = 'Exponential Smoothing')

# ======================================================================================================================
# ARIMA
# ======================================================================================================================

model_arima = Arima(DL)
model_arima.fit(seasonal=True, seasonality=7)
forecast_arima = model_arima.predict(start=model_arima.test.index[0], end=model_arima.test.index[-1])
eval_arima = EvaluateModel(model_arima.test[model_arima.target], forecast_arima)
print('RMSE for ARIMA model: ', eval_arima.rmse())
eval_arima.plot(label = 'ARIMA')

# ======================================================================================================================
# Prophet
# ======================================================================================================================

model_pr = FBProphet(df)
model_pr.fit()
forecast_pr = model_pr.predict(start=model_pr.test.index[0], end=model_pr.test.index[-1])

eval_pr = EvaluateModel(model_pr.test, forecast_pr)
print('RMSE for Prophet model: ', eval_pr.rmse())
eval_pr.plot(label = 'Prophet')

# ======================================================================================================================
# Random Forest
# ======================================================================================================================

FE = FeatureEngineering(DL)
FE.generate_lags(features_to_lag=['Customers'], lags=[1,12])
FE.split_features_target(type='train')

model_rf = RandomForest(FE)
model_rf.fit(trend='additive', seasonality=12)
X_test, y_test = FE.split_features_target(type='test')
X_test = X_test[model_rf.selected_features]
forecast_rf = model_rf.predict(X_test)
forecast_rf.set_index(keys=model_rf.test.index, inplace = True)

eval_rf = EvaluateModel(model_rf.test[model_rf.target], forecast_rf)
print('RMSE for Random Forest model: ', eval_rf.rmse())
eval_rf.plot(label = 'Random Forest')


# ======================================================================================================================
# Extra-tree regressor
# ======================================================================================================================

FE = FeatureEngineering(DL)
FE.generate_lags(features_to_lag=['Customers'], lags=[1,12])
FE.split_features_target(type='train')

model_xtree = ExtraTree(FE)
model_xtree.fit(trend='additive', seasonality=12)
X_test, y_test = FE.split_features_target(type='test')
X_test = X_test[model_xtree.selected_features]
forecast_xtree = model_xtree.predict(X_test)
forecast_xtree.set_index(keys=model_xtree.test.index, inplace = True)

eval_xtree = EvaluateModel(model_xtree.test[model_xtree.target], forecast_xtree)
print('RMSE for Extra-tree regressor: ', eval_xtree.rmse())
eval_xtree.plot(label = 'Extra-tree regressor')

# ======================================================================================================================
# XGBoost
# ======================================================================================================================

FE = FeatureEngineering(DL)
FE.generate_lags(features_to_lag=['Customers'], lags=[1,12])
FE.split_features_target(type='train')

model_xgb = XGBoost(FE)
model_xgb.fit(trend='additive', seasonality=12)
X_test, y_test = FE.split_features_target(type='test')
X_test = X_test[model_xgb.selected_features]
forecast_xgb = model_xgb.predict(X_test.to_numpy())
forecast_xgb.set_index(keys=model_xgb.test.index, inplace = True)

eval_xgb = EvaluateModel(model_xgb.test[model_xgb.target], forecast_xgb)
print('RMSE for XGBoost model: ', eval_xgb.rmse())
eval_xgb.plot(label = 'XGBoost')


# ======================================================================================================================
# Support Vector Regressor (SVR)
# ======================================================================================================================

FE = FeatureEngineering(DL)
FE.generate_lags(features_to_lag=['Customers'], lags=[1,12])
FE.split_features_target(type='train')

model_svr = SupportVectorRegressor(FE)
model_svr.fit(trend='additive', seasonality=12)
X_test, y_test = FE.split_features_target(type='test')
X_test = X_test[model_svr.selected_features]
forecast_svr = model_svr.predict(X_test.to_numpy())
forecast_svr.set_index(keys=model_svr.test.index, inplace = True)

eval_svr = EvaluateModel(model_svr.test[model_svr.target], forecast_svr)
print('RMSE for Support Vector Regressor model: ', eval_svr.rmse())
eval_svr.plot(label = 'Support Vector Regressor')


# ======================================================================================================================
# Ensemble using average weights
# ======================================================================================================================
model_en = EnsembleModels(model_es, model_arima, model_pr)
forecast_en = model_en.predict(start=model_pr.test.index[0], end=model_pr.test.index[-1], mode = 'average')

eval_en = EvaluateModel(model_pr.test, forecast_en)
print('RMSE for ensemble using average model: ', eval_en.rmse())
eval_en.plot(label = 'Average ensemble')

# ======================================================================================================================
# Ensemble using optimized weights
# ======================================================================================================================
model_en = EnsembleModels(model_es, model_arima, model_pr)
forecast_en = model_en.predict(start=model_pr.test.index[0], end=model_pr.test.index[-1], mode = 'optimized')

eval_en = EvaluateModel(model_pr.test, forecast_en)
print('RMSE for ensemble using optimized weights model: ', eval_en.rmse())
eval_en.plot(label = 'Optimized ensemble')