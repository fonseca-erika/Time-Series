from models import Exponential_Smoothing, Arima, EnsembleModels, FBProphet, RandomForest, XGBoost, SupportVectorRegressor
import pandas as pd

def predict_exogeneous(DL, model):

    df_exog = pd.DataFrame(columns=['Date'])

    for c in DL.df.select_dtypes('number').columns:

        DL.set_target(c)

        if model == 'Exponential Smoothing':
            model = Exponential_Smoothing(DL)
        elif model == 'ARIMA':
            model = Arima(DL)

        forecast = model.fit_predict(start=model.test.index[0], end=model.test.index[-1])

        if df_exog.empty == True:
            df_exog['Date'] = forecast.index
            df_exog[c] = forecast.values
        else:
            if forecast is None:
                print('---> Variable {0}. Please check the data!'.format(c))
            else:
                df_exog[c] = forecast.values

    return df_exog