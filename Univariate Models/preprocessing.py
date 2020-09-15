from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split


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
