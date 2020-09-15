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
        return np.sqrt(mean_squared_error(self.df_actual, self.df_forecast))

    def plot(self, label = 'Forecast values'):
        plt.figure(figsize=(15, 5))
        plt.title("Forecast vs Actual")
        plt.plot(self.df_actual, label="Actual values")
        plt.plot(self.df_forecast, label=label)
        plt.legend(loc="upper left")
        plt.plot()
        plt.show()

