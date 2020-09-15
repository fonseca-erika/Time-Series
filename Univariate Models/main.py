from dataload import DataLoad
from evaluation import EvaluateModel
from preprocessing import DataSplit
from models import Exponential_Smoothing, AutoArima, EnsembleModels, FBProphet
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.simplefilter("ignore")
from datapreparation import DataVisualization

print('Time Series model composition')

# Loading Data
DL = DataLoad()
df = DL.load_csv('data/ross_train.csv')
df = df[df['Store'] == 1]
df = df[['Date', 'Sales']]
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

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
model_es = Exponential_Smoothing(df)
model_es.fit(trend_effect='add', damped=True, seasonal_effect='add', seasonality=7, use_boxcox=False, frequency='D')
forecast_es = model_es.predict(start=model_es.test.index[0], end=model_es.test.index[-1])

eval_es = EvaluateModel(model_es.test, forecast_es)
print('RMSE for exponential smoothing model: ', eval_es.rmse())
eval_es.plot(label = 'Exponential Smoothing')

# Using cross-validation to anlayse the error
# model_es.fit_cv(trend_effect='add', damped=True, seasonal_effect='add', seasonality=7, use_boxcox=False, frequency='D',
#                 k=10)
# forecast_es = model_es.predict(start=model_es.test.index[0], end=model_es.test.index[-1])
# eval_es = EvaluateModel(model_es.test, forecast_es)
# eval_es.plot(label = 'Exponential Smoothing')

# ======================================================================================================================
# Auto ARIMA
# ======================================================================================================================

model_arima = AutoArima(df)
model_arima.fit(seasonal=True, seasonality=7)
forecast_arima = model_arima.predict(start=model_arima.test.index[0], end=model_arima.test.index[-1]) #n = model_arima.test.shape[0])
eval_arima = EvaluateModel(model_arima.test, forecast_arima)
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