from pmdarima import auto_arima
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from fbprophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import abc

from sklearn.model_selection import TimeSeriesSplit  # time Series cross-validator
from sklearn.feature_selection import RFECV  # feature selection
from sklearn.model_selection import GridSearchCV  # hyperparameter tunning
from sklearn.model_selection import ParameterGrid

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from dataload import DataLoad
from evaluation import EvaluateModel
from preprocessing import DataSplit, FeatureEngineering


class TimeSeriesModel(object, metaclass=abc.ABCMeta):

    def __init__(self, DataObject, train_size=.8):
        # Dividing into train and test set
        if type(DataObject) == DataLoad or type(DataObject) == FeatureEngineering:
            self.train, self.test = DataSplit(train_size).sequential_split(DataObject.df)
            self.target = DataObject.target
            self.date = DataObject.date
            self.features = DataObject.features
            self.frequency = DataObject.frequency
            self.df = DataObject.df
            self.set_name()
        else:
            print("ERROR: DataObject must be of class DataLoad or FeatureEngineering")

        if type(DataObject) == FeatureEngineering:
            self.X = DataObject.X
            self.y = DataObject.y

    def feature_selection(self, model, k=10, show_details=True):
        '''
         model: a supervised learning estimator with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute.
         k: number of folders for cross validation
        '''
        rfecv = RFECV(estimator=model, step=1, cv=TimeSeriesSplit(k))
        rfecv.fit(self.X.to_numpy(), self.y.to_numpy())

        self.selected_X = rfecv.transform(self.X)
        self.selected_features = self.X.columns[rfecv.get_support(1)]

        if show_details == True:
            print("Selected features :", self.selected_features)

            # Plot number of features VS. cross-validation scores
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.show()

        return self.selected_X

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

    def calculate_period(self, start, end):
        df_range = pd.date_range(start=start,
                                 end=end, freq=self.frequency)
        n = df_range.shape[0]
        return n

    def fit_predict(self, start, end):

        self.start = start
        self.end = end
        self.forecast = None

        try:
            self.fit()
            self.forecast = self.predict(start=self.start, end=self.end)
        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class Exponential_Smoothing(TimeSeriesModel):

    def set_name(self):
        self.name = 'Exponential Smoothing'

    def GridSearch_CV(self, *params):
        """
        Iterating over folds, train model on each fold's training set,
        forecast and calculate error on each fold's test set.
        """

        k = 10
        errors = []
        params = params[0]

        t = params['trend']
        d = params['damped_trend']
        s = params['seasonal']
        p = params['seasonal_periods']
        u = params['use_boxcox']

        tscv = TimeSeriesSplit(n_splits=k)

        for train, test in tscv.split(self.train[self.target]):

            error = np.Inf

            try:
                mdl = ExponentialSmoothing(endog=self.train[self.target][train], trend=t, damped_trend=d, seasonal=s,
                                           seasonal_periods=p, initialization_method='legacy-heuristic').fit(
                    use_boxcox=u)

                predictions = mdl.predict(start=self.train[self.target][test].index[0],
                                          end=self.train[self.target][test].index[-1])
                actual = self.train[self.target][test]
                error = mean_squared_error(actual, predictions)

            finally:
                errors.append(error)
                return np.mean(errors)

    def fit(self):

        params = {'trend': ['add', 'mul'],
                  'damped_trend': [True, False],
                  'seasonal': ['add', 'mul'],
                  'seasonal_periods': [4, 12],
                  'use_boxcox': [False, True]}

        param_grid = list(ParameterGrid(params))

        score = []
        for p in param_grid:
            score.append(self.GridSearch_CV(p))

        best_params_ = param_grid[np.argmin(score)]

        self.trend = best_params_['trend']
        self.damped = best_params_['damped_trend']
        self.seasonal = best_params_['seasonal']
        self.seasonality = best_params_['seasonal_periods']
        self.use_boxcox = best_params_['use_boxcox']

        self.model = None

        try:
            self.model = ExponentialSmoothing(endog=self.train[self.target], trend=self.trend,
                                              damped_trend=self.damped, seasonal=self.seasonal,
                                              seasonal_periods=self.seasonality,
                                              initialization_method='legacy-heuristic').fit(use_boxcox=self.use_boxcox)

        except:
            print('ERROR: Could not generate model for time series using {0}.'.format(self.name))

        return self.model

    def predict(self, start, end):

        self.start = start
        self.end = end
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.model.predict(start=self.start, end=self.end))
        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class Arima(TimeSeriesModel):

    def set_name(self):
        self.name = 'ARIMA'

    def GridSearch_CV(self, *params):
        """
        Iterating over folds, train model on each fold's training set,
        forecast and calculate error on each fold's test set.
        """
        k = 5
        errors = []
        params = params[0]

        s = params['seasonal']
        m = params['m']

        tscv = TimeSeriesSplit(n_splits=k)

        for train, test in tscv.split(self.train[self.target]):
            error = np.Inf

            try:
                mdl = auto_arima(self.train[self.target][train], seasonal=s, m=m)

                n = self.calculate_period(start=self.train[self.target][test].index[0],
                                          end=self.train[self.target][test].index[-1])
                predictions = mdl.predict(n)
                actual = self.train[self.target][test]

                error = mean_squared_error(actual, predictions)

            finally:
                errors.append(error)

        return np.mean(errors)

    def fit(self):

        params = {'seasonal': [True],
                  'm': [4, 12]}

        param_grid = list(ParameterGrid(params))
        param_grid.append({'seasonal': False, 'm': 1})

        score = []
        for p in param_grid:
            score.append(self.GridSearch_CV(p))

        best_params_ = param_grid[np.argmin(score)]

        self.seasonal = best_params_['seasonal']
        self.seasonality = best_params_['m']

        # try:
        self.model = auto_arima(self.train[self.target], seasonal=self.seasonal, m=self.seasonality)

        # print(self.model.get_params())
        # except:
        # print('ERROR: Could not generate a model for time series using {0}.'.format(self.name))

        return self.model

    def predict(self, start, end):

        self.start = start
        self.end = end

        n = self.calculate_period(start=self.start, end=self.end)
        self.forecast = None

        # try:
        self.forecast = pd.DataFrame(self.model.predict(n))
        self.forecast['Date'] = pd.date_range(start=self.start, end=self.end)
        self.forecast.set_index('Date', inplace=True)
        # except:
        #    print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class FBProphet(TimeSeriesModel):

    def set_name(self):
        self.name = 'Prophet'

    def GridSearch_CV(self, *params):
        """
        Iterating over folds, train model on each fold's training set,
        forecast and calculate error on each fold's test set.
        """
        k = 10
        errors = []
        params = params[0]

        g = params['growth']
        y = params['yearly_seasonality']
        w = params['weekly_seasonality']
        d = params['daily_seasonality']
        s = params['seasonality_mode']

        tscv = TimeSeriesSplit(n_splits=k)

        for train, test in tscv.split(self.train[self.target]):
            error = np.Inf

            try:

                df_prophet = self.train.copy()
                df_prophet.reset_index(inplace=True)
                df_prophet.rename(columns={self.date: 'ds', self.target: 'y'}, inplace=True)

                mdl = Prophet(growth=g, yearly_seasonality=y, weekly_seasonality=w, daily_seasonality=d,
                              seasonality_mode=s)
                mdl = mdl.fit(df_prophet)

                n = self.calculate_period(start=self.train[self.target][test].index[0],
                                          end=self.train[self.target][test].index[-1])

                predictions = mdl.make_future_dataframe(periods=n,
                                                        freq=self.frequency)  # create the dataframe with the future timestamp
                predictions = mdl.predict(predictions)  # runs the forecast
                predictions[self.target] = predictions.yhat[-n:]  # save the forecast on the dataframe
                # The Prophet will do the forecast for the whole period, so we take only the predictions that represent the interval
                predictions = predictions[-n:]
                actual = self.train[self.target][test]

                error = mean_squared_error(actual, predictions[self.target])

            finally:
                errors.append(error)

        return np.mean(errors)

    def fit(self):

        params = {'growth': ['linear'],
                  'yearly_seasonality': ['auto'],
                  'weekly_seasonality': ['auto'],
                  'daily_seasonality': ['auto'],
                  'seasonality_mode': ['additive', 'multiplicative']}

        param_grid = list(ParameterGrid(params))

        score = []
        for p in param_grid:
            score.append(self.GridSearch_CV(p))

        best_params_ = param_grid[np.argmin(score)]

        self.growth = best_params_['growth']
        self.yearly_seasonality = best_params_['yearly_seasonality']
        self.weekly_seasonality = best_params_['weekly_seasonality']
        self.daily_seasonality = best_params_['daily_seasonality']
        self.seasonality_mode = best_params_['seasonality_mode']

        # Renaming columns for the standard of Prophet library
        self.train['ds'] = self.train.index
        self.train['y'] = self.train[self.target]

        self.model = Prophet(growth=self.growth, yearly_seasonality=self.yearly_seasonality,
                             weekly_seasonality=self.weekly_seasonality, daily_seasonality=self.daily_seasonality,
                             seasonality_mode=self.seasonality_mode)

        self.model = self.model.fit(self.train)

        return self.model

    def predict(self, start, end):

        self.start = start
        self.end = end

        n = self.calculate_period(start=self.start, end=self.end)

        self.forecast = None

        try:
            self.forecast = self.model.make_future_dataframe(periods=n,
                                                             freq=self.frequency)  # create the dataframe with the future timestamp
            forecast = self.model.predict(self.forecast)  # runs the forecast
            self.forecast[self.target] = forecast.yhat  # save the forecast on the dataframe
            self.forecast.rename(columns={'ds': 'Date'}, inplace=True)
            self.forecast.set_index('Date', inplace=True)
            # The Prophet will do the forecast for the whole period, so we take only the predictions that represent the interval
            self.forecast = self.forecast[-n:]

        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

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


class RandomForest(TimeSeriesModel):

    def set_name(self):
        self.name = 'Random Forest'

    def fit(self, trend, seasonality):

        self.fit = None
        self.trend = trend
        self.seasonality = seasonality

        self.lags = [1, seasonality]

        try:
            rf = RandomForestRegressor()
            self.feature_selection(model=rf)

            param_grid = {
                'n_estimators': [200, 500, 1000],
                'max_depth': [2, 4, 6, 8]
            }

            model = GridSearchCV(estimator=rf, param_grid=param_grid,
                                 cv=TimeSeriesSplit(n_splits=10).get_n_splits([self.selected_X, self.y.to_numpy()]))

            self.fit = model.fit(self.selected_X, self.y.to_numpy())

        except:
            print('ERROR: Could not run random forest! Please check your input data. \n')

        return self.fit

    def predict(self, inputs):

        self.inputs = inputs
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(self.inputs))

        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class XGBoost(TimeSeriesModel):

    def set_name(self):
        self.name = 'XGBoost'

    def fit(self, trend, seasonality):

        self.fit = None
        self.trend = trend
        self.seasonality = seasonality

        self.lags = [1, seasonality]

        try:
            xgb = XGBRegressor()
            self.feature_selection(model=xgb)

            param_grid = {
                'n_estimators': [200, 500, 1000],
                'max_depth': [2, 4, 6, 8],
                'learning_rate': [0.001, 0.05, 0.1]
            }

            model = GridSearchCV(estimator=xgb, param_grid=param_grid)
            self.fit = model.fit(self.selected_X, self.y.to_numpy())

        except:
            print('ERROR: Could not run XGBoost! Please check your input data. \n')

        return self.fit

    def predict(self, inputs):

        self.inputs = inputs
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(self.inputs))

        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class ExtraTree(TimeSeriesModel):

    def set_name(self):
        self.name = 'Extra Tree Regressor'

    def fit(self, trend, seasonality):

        self.fit = None
        self.trend = trend
        self.seasonality = seasonality

        self.lags = [1, seasonality]

        # try:
        xtree = ExtraTreeRegressor()
        self.feature_selection(model=xtree)

        param_grid = {
            'max_depth': [2, 4, 6, 8]
        }

        model = GridSearchCV(estimator=xtree, param_grid=param_grid)
        self.fit = model.fit(self.selected_X, self.y.to_numpy())

        # except:
        #  print('ERROR: Could not run Support Vector Regressor! Please check your input data. \n')

        return self.fit

    def predict(self, inputs):

        self.inputs = inputs
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(self.inputs))

        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast


class SupportVectorRegressor(TimeSeriesModel):

    def set_name(self):
        self.name = 'Support Vector Regressor'

    def fit(self, trend, seasonality):

        self.fit = None
        self.trend = trend
        self.seasonality = seasonality

        self.lags = [1, seasonality]

        # try:
        svr = SVR(kernel='linear')
        self.feature_selection(model=svr)

        param_grid = {
            'C': [.01, .1, 1],
            'epsilon': [.001, .01, 0.1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4]
        }

        model = GridSearchCV(estimator=svr, param_grid=param_grid)
        self.fit = model.fit(self.selected_X, self.y.to_numpy())

        # except:
        #  print('ERROR: Could not run Support Vector Regressor! Please check your input data. \n')

        return self.fit

    def predict(self, inputs):

        self.inputs = inputs
        self.forecast = None

        try:
            self.forecast = pd.DataFrame(self.fit.predict(self.inputs))

        except:
            print('ERROR: Could not forecast this time series using {0}.'.format(self.name))

        return self.forecast
