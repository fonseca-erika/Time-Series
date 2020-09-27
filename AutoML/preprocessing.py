from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from dataload import DataLoad


class DataScaler(object):

    def __init__(self, type='RobustScaler'):

        if type == 'RobustScaler':
            self.scaler = RobustScaler()
        elif type == 'StandardScaler':
            self.scaler = StandardScaler()
        elif type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()

    def transform(self, train_df, test_df):

        try:

            self.train = self.scaler.fit_transform(train_df.values)

            if (test_df is not None):
                self.test = self.scaler.transform(test_df.values)
            else:
                self.test = None

        except:

            print('ERROR: could not perform transformation. Please check if your input is a valid dataframe')
            self.train = None
            self.test = None

        return self.train, self.test

    def reverse(self, df):

        return self.scaler.inverse_transform(df.values)


class DataSplit(object):

    def __init__(self, train_size=.8):

        self.train_size = train_size

    def sequential_split(self, df):

        if (self.train_size < 0 or self.train_size > 1):
            print('ERROR: invalid train_size. Value should be between 0 and 1.')
        elif len(df) == 0:
            print('ERROR: invalid dataframe. Check the input data')
        else:

            n = int(len(df) * self.train_size)
            # Store the raw data.
            self.train_df = df[0:n]
            self.test_df = df[n:]

        return self.train_df, self.test_df

    def random_split(self, df):

        if (self.train_size < 0 or self.train_size > 1):
            print('ERROR: invalid train_size. Value should be between 0 and 1.')
        elif len(df) == 0:
            print('ERROR: invalid dataframe. Check the input data')
        else:

            self.train_df, self.test_df = train_test_split(df, train_size=self.train_size)

        return self.train_df, self.test_df


class DataClean(object):

    def __init__(self, df):
        self.df = df

    def delete_columns(self):
        pass

    def delete_rows(self):
        pass


class FeatureEngineering(object):

    def __init__(self, DataObject):
        if type(DataObject) == DataLoad:
            self.df = DataObject.df
            self.train, self.test = DataSplit(.8).sequential_split(self.df)
            self.target = DataObject.target
            self.features = DataObject.features
        else:
            print("ERROR: DataObject must be of class DataLoad")

    def generate_lags(self, features_to_lag, lags):

        self.lag_df = None

        if type(features_to_lag) != list or type(lags) != list:
            print('ERROR: parameters features_to_lag and lags must be of list type')
        else:
            if features_to_lag == None or features_to_lag == []:
                self.features_to_lag = self.features
            else:
                self.features_to_lag = features_to_lag

            if lags == '' or lags == None:
                self.lags = [1]
            else:
                self.lags = lags

            self.lag_df = pd.DataFrame()
            for feature in self.features_to_lag:
                for lag in self.lags:
                    self.lag_df[feature + '-t-' + str(lag)] = self.df[feature].shift(lag)

            # We miss the information for the lagging, so we have to consider it
            self.lag_df = self.lag_df[max(lags):]
            self.lag_df = self.df[max(lags):].join(self.lag_df, how='inner')

            features = list(self.lag_df.columns)
            features.remove(self.target)
            self.features = features

        return self.lag_df

    def split_features_target(self, type='full'):

        self.X = self.lag_df[self.features]
        self.y = self.lag_df[self.target]
        if type == 'train':
            self.X = self.X[:self.train.shape[0]]
            self.y = self.y[:self.train.shape[0]]
        elif type == 'test':
            self.X = self.X[-self.test.shape[0]:]
            self.y = self.y[-self.test.shape[0]:]
        else:
            print("ERROR: Invalid type_, values allowed are full, train and test.")

        return self.X, self.y

    def create_features(self):
        """
        Creates time series features from datetime index
        """
        self.df['dayofweek'] = self.df.index.dt.dayofweek
        self.df['quarter'] = self.df.index.dt.quarter
        self.df['month'] = self.df.index.dt.month
        self.df['year'] = self.df.index.dt.year
        self.df['dayofyear'] = self.df.index.dt.dayofyear
        self.df['dayofmonth'] = self.df.index.dt.day
        self.df['weekofyear'] = self.df.index.dt.weekofyear

        return self.df

    def one_hot_enconding(self):

        pass

    def fillna(self, type='interpolate'):

        '''
        :param

            type: method to use for filling holes in reindexed ['mean', 'mode', 'bfill', 'ffill', 'interpolate']
            - ffill: propagate last valid observation forward to next valid
            - bfill: use next valid observation to fill gap

        :return:

            Dataframe
        '''
        self.type = type

        if self.df.isna().sum().sum() > 0:

            for col in self.df.columns[self.df.dtypes == 'category' or self.df.dtypes == 'object']:
                self.df[col].fillna(self.df[col].value_counts().idxmax())  # fill with the most frequent value

            if self.type == 'mean':
                self.df.fillna(self.df.mean(), inplace = True)
            elif self.type == 'median':
                self.df.fillna(self.df.median(), inplace=True)
            elif self.type == 'interpolate':
                self.df.interpolate(inplace=True)
            elif self.type == 'bfill' or self.type == 'ffill':
                self.df.fillna(method=self.type, inplace=True)

        return self.df

