import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


class EvaluateModel(object):

    def __init__(self, df_actual, df_forecast):
        self.df_actual = df_actual
        self.df_forecast = df_forecast
        print("actual")
        print(self.df_actual)
        print('forecast')
        print(self.df_forecast)

    def rmse(self):

        error = None
        if self.df_actual is None and self.df_forecast is None:
            print('ERROR: invalid input for actual values and predicted values!')
        elif self.df_forecast is None:
            print('ERROR: invalid input for predicted values!')
        elif self.df_actual is None:
            print('ERROR: invalid input for actual values!')
        else:
            error = np.sqrt(mean_squared_error(self.df_actual, self.df_forecast))

        return error

    def plot(self, label='Forecast values'):

        if self.df_actual is None or self.df_forecast is None:
            print('ERROR: could not plot due to invalid input data for actual values and/or predicted values!')
        else:
            plt.figure(figsize=(15, 5))
            plt.title("Forecast vs Actual")
            plt.plot(self.df_actual, label="Actual values")
            plt.plot(self.df_forecast, label=label)
            plt.legend(loc="upper left")
            plt.plot()
            plt.show()
