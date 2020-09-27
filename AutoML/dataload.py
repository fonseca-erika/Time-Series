import pandas as pd
import pyodbc
import os


class DataLoad(object):

    def __init__(self):
        self.df = None

    def load_csv(self, file_path, date, target, show_details=True):

        self.date = date
        self.target = target

        if os.path.isdir(file_path):
            print('ERROR: informed directory instead of file path.')
        else:
            try:
                self.df = pd.read_csv(file_path, low_memory=False)
                if show_details:
                    print(self.df.info())
                    print(self.df.describe())

                self.df[date] = pd.to_datetime(self.df[date])
                self.df.set_index(date, inplace=True)
                self.df.sort_index(inplace=True)
                self.frequency = self.df.index.inferred_freq

                if self.target in self.df.columns:
                    features = list(self.df.columns)
                    self.features = features.remove(self.target)
                else:
                    print("ERROR: '{0}' field not found on the dataset.".format(self.target))

            except FileNotFoundError:
                print('ERROR: File not found!')
            except pd.io.parsers.ParserError:
                print('ERROR: Invalid file. Check if format is CSV or file is corrupted!')
            except KeyError as k:
                print('ERROR: {0} field not found on the dataset.'.format(k))
            except ValueError:
                print('ERROR: invalid format for field {0}.'.format(self.date))
            except:
                print('ERROR: Could not open the file!')

        return self.df

    def diagnosis(self):

        self.status = 'OK'
        if self.df.isna().sum().sum() > 0:
            self.status = 'NOK'

        return self.status


    def set_target(self, target):

        if target in self.df.columns:
            self.target = target
            features = list(self.df.columns)
            self.features = features.remove(self.target)
        else:
            print("ERROR: '{0}' field not found on the dataset.".format(target))

        return self.df

    @staticmethod
    def __connect_RDBMS(db_type, server, db_name):

        sql_conn = None

        if db_type == 'SQL Server':
            connectionstring = 'DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' + server + ';DATABASE=' + db_name + ';Trusted_Connection=yes'
            try:
                sql_conn = pyodbc.connect(connectionstring)

            except:
                print('ERROR: failed to connect to database! Try to check the connection parameters.')
        else:
            print('ERROR: Database type not supported!')

        return sql_conn

    def load_SQL(self, db_type, server, db_name, query_string):

        sql_conn = DataLoad.__connect_RDBMS(db_type, server, db_name)

        if not (sql_conn is None):
            try:
                self.df = pd.read_sql(query_string, sql_conn)
            except:
                print('ERROR: Query failed!')

        return self.df
