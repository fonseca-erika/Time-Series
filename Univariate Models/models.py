from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from numpy import mean, std
from sklearn.model_selection import TimeSeriesSplit
import abc
from datetime import date
from fbprophet import Prophet
import numpy as np
from scipy.optimize import minimize

from evaluation import EvaluateModel
from preprocessing import DataSplit


class TimeSeriesModel(object, metaclass=abc.ABCMeta):

    def __init__(self, df, train_size=.8):
        # Dividing into train and test set
        self.train, self.test = DataSplit(train_size).sequential_split(df)

    # Using abstract I oblige the implementation in the inheritance hierarchy
    @abc.abstractmethod
    def set_name(self):
        pass

    @abc.abstractmethod
    def fit(self, **params):
        pass

    @abc.abstractmethod
    def predict(self, **params):
        pass


class Exponential_Smoothing(TimeSeriesModel):

    def set_name(self):
        self.name = 'Exponential Smoothing'

    def fit(self, trend_effect='add', damped=True, seasonal_effect='add', seasonality=12, frequency='D',
            use_boxcox=True):

        self.trend = trend_effect
        self.damped = damped
        self.seasonal = seasonal_effect
        self.seasonality = seasonality
        self.frequency = frequency
        self.use_boxcox = use_boxcox

        self.fit = None

        if self.trend not in ['add', 'mult', None]:
            print('ERROR: Invalid value for trend effect! Allowed values are: add, mult or None')
        elif self.seasonal not in ['add', 'mult', None]:
            print('ERROR: Invalid value for trend effect! Allowed values are: add, mult or None')
        elif self.use_boxcox not in [True, False]:
            print('ERROR: Invalid value for use_boxcox! Allowed values are: True or False')
        elif self.seasonality < 0 or self.seasonality > self.train.shape[0]:
            print('ERROR: Invalid value for seasonality!')
        elif not isinstance(self.seasonality, int):
            print('ERROR: Seasonality must be an integer!')
        else:
            try:

                self.fit = ExponentialSmoothing(endog=self.train, trend=self.trend, damped_trend=self.damped,
                                                seasonal=self.seasonal, seasonal_periods=self.seasonality,
                                                initialization_method='legacy-heuristic', freq=self.frequency).fit()

            except:
                print('ERROR: Could not run exponential smoothing! Please check your input data. \n')

        return self.fit

    def predict(self, start, end):

        self.start = start
        self.end = end
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(self.start, self.end))
        except:
            print('ERROR: Could not forecast using Exponential Smoothing model this time series.')

        return self.forecast

    def fit_cv(self, trend_effect='add', damped=True, seasonal_effect='add', seasonality=12, frequency='D',
               use_boxcox=True, k=10):

        self.trend = trend_effect
        self.damped = damped
        self.seasonal = seasonal_effect
        self.seasonality = seasonality
        self.frequency = frequency
        self.use_boxcox = use_boxcox
        self.fit = None
        rmse_stats = []

        if self.trend not in ['add', 'mult', None]:
            print('ERROR: Invalid value for trend effect! Allowed values are: add, mult or None')
        elif self.seasonal not in ['add', 'mult', None]:
            print('ERROR: Invalid value for trend effect! Allowed values are: add, mult or None')
        elif self.use_boxcox not in [True, False]:
            print('ERROR: Invalid value for use_boxcox! Allowed values are: True or False')
        elif self.seasonality < 0 or self.seasonality > self.train.shape[0]:
            print('ERROR: Invalid value for seasonality!')
        elif not isinstance(self.seasonality, int):
            print('ERROR: Seasonality must be an integer!')
        else:

            self.forecast = None
            try:
                tscv = TimeSeriesSplit(n_splits=k)
                for train_index, test_index in tscv.split(self.train):  # The split function returns only indexes
                    train = self.train.iloc[train_index]
                    test = self.train.iloc[test_index]
                    self.start = test.index[0]
                    self.end = test.index[-1]
                    self.fit = ExponentialSmoothing(endog=train, trend=self.trend, damped_trend=self.damped,
                                                    seasonal=self.seasonal, seasonal_periods=self.seasonality,
                                                    initialization_method='legacy-heuristic', freq=self.frequency).fit()
                    self.forecast = pd.DataFrame(self.fit.predict(self.start, self.end))
                    rmse_stats.append(EvaluateModel(test, self.forecast).rmse())

                print("RSME for exponential smoothing computed using cross validation: %0.2f (+/- %0.2f)" % (
                    mean(rmse_stats), std(rmse_stats) * 2))

                # Select the best model considering the cross validation concept

                self.fit = ExponentialSmoothing(endog=self.train, trend=self.trend, damped_trend=self.damped,
                                                seasonal=self.seasonal, seasonal_periods=self.seasonality,
                                                initialization_method='legacy-heuristic', freq=self.frequency).fit()

            except:
                print('ERROR: Could not run exponential smoothing! Please check your input data. \n')  # ) + ex.message)
        return self.fit


class AutoArima(TimeSeriesModel):

    def set_name(self):
        self.name = 'ARIMA'

    def fit(self, seasonal, seasonality):

        self.seasonal = seasonal
        if not isinstance(seasonality, int):
            self.fit = None
            print('ERROR: Invalid value for seasonality')
        else:
            self.seasonality = seasonality
            self.fit = auto_arima(self.train, seasonal=self.seasonal, m=self.seasonality)

        return self.fit

    def predict(self, start, end):

        self.start = start
        self.end = end

        n = abs(self.end - self.start).days + 1
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(n))
            self.forecast['Date'] = pd.date_range(start=self.start, end=self.end)
            self.forecast.set_index('Date', inplace=True)
        except:
            print('ERROR: Could not forecast using ARIMA model this time series.')

        return self.forecast


class FBProphet(TimeSeriesModel):

    def set_name(self):
        self.name = 'Prophet'

    def fit(self, growth='linear', yearly_seasonality='auto', weekly_seasonality='auto',
            daily_seasonality='auto', holidays=None, seasonality_mode='additive'):

        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode

        # Renaming columns for the standard of Prophet library
        self.train['ds'] = self.train.index
        self.train['y'] = self.train['Sales']

        self.model = Prophet(growth=self.growth, yearly_seasonality=self.yearly_seasonality,
                             weekly_seasonality=self.yearly_seasonality, daily_seasonality=self.daily_seasonality,
                             holidays=self.holidays, seasonality_mode=self.seasonality_mode)
        self.fit = self.model.fit(self.train)

        return self.fit

    def predict(self, start, end):

        self.start = start
        self.end = end

        n = abs(self.end - self.start).days + 1
        self.forecast = None

        try:
            self.forecast = self.model.make_future_dataframe(
                periods=n)  # create the dataframe with the future timestamp
            forecast = self.model.predict(self.forecast)  # runs the forecast
            self.forecast['Sales'] = forecast.yhat  # save the forecast on the dataframe
            self.forecast.rename(columns={'ds': 'Date'}, inplace=True)
            self.forecast.set_index('Date', inplace=True)
            # The Prophet will do the forecast for the whole period, so we take only the predictions that represent the interval
            self.forecast = self.forecast[-n:]

        except:
            print('ERROR: Could not forecast using Prophet model this time series.')

        return self.forecast


class EnsembleModels(object):

    def set_name(self):
        self.name = 'Ensemble'

    def __init__(self, *models_list):

        self.models = list(models_list)

    def predict(self, start, end, mode='average'):

        self.start = start
        self.end = end
        self.mode = mode
        self.len = len(self.models)
        self.weights = np.zeros(self.len)
        self.weights.fill(1 / self.len)

        for idx, model in enumerate(self.models):
            if idx == 0:
                forecast = model.predict(start=self.start, end=self.end)
            else:
                forecast['model_' + str(idx)] = model.predict(start=self.start, end=self.end)

        if self.mode == 'average':

            self.forecast = (forecast * self.weights).sum(axis=1)
        elif self.mode == 'optimized':

            # Using as criteria the RSME to choose the best weights
            fun = lambda w: EvaluateModel(model.test, (forecast * w).sum(axis=1)).rmse()  # objective function
            cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})  # constraint
            res = minimize(fun, self.weights, method='SLSQP', constraints=cons)
            self.weights = res.x
            self.forecast = (forecast * self.weights).sum(axis=1)

            print('weights', self.weights)
        else:
            print('WARNING: Invalid mode for generating the ensemble')

        return self.forecast

