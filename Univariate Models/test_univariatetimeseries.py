import unittest
from dataload import DataLoad
import pandas as pd


class TestDataload(unittest.TestCase):

    def test_file_exists(self):
        # A dataframe must be returned when the path exists
        print('testing valid file...')
        DL = DataLoad()
        df = DL.load_csv('data/ross_train.csv', show_details=False)
        self.assertIsInstance(df, pd.DataFrame)

    def test_file_does_not_exist(self):
        print('testing inexistent file...')
        DL = DataLoad()
        df = DL.load_csv('data/ross_trai.csv')
        self.assertEqual(df, None)

    def test_corrupted_file(self):
        print('testing corrupted file...')
        DL = DataLoad()
        df = DL.load_csv('data/corrupted.csv')
        self.assertEqual(df, None)

    def test_dir(self):
        print('testing dir instead of file...')
        DL = DataLoad()
        df = DL.load_csv('data')
        self.assertEqual(df, None)

    def test_invalid_format(self):
        print('testing invalid file format - not CSV...')
        DL = DataLoad()
        df = DL.load_csv('data/transportation.json')
        self.assertEqual(df, None)


if __name__ == '__main__':
    unittest.main()
