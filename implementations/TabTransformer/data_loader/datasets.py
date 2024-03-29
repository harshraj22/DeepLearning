import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from pprint import pprint
from tqdm import tqdm
import numpy as np
import unittest

sys.path.append('..')

from utils.utils import CategoryEncoder


class BlastcharDataset(Dataset):
    """ Class for efficiently loading the Blastchar dataset.
    src: https://www.kaggle.com/blastchar/telco-customer-churn
    """

    CATEGORICAL_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    CONTINIOUS_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']

    def __init__(self, dataset_path):
        super(BlastcharDataset, self).__init__()

        self.df = pd.read_csv(dataset_path)
        # encode the labels
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        self.categoricals = self.df.loc[:,self.CATEGORICAL_COLUMNS]
        self.encoder = CategoryEncoder([self.categoricals[header].values.tolist() for header in self.CATEGORICAL_COLUMNS])

        # some of the values in dataset were missing, causing faults while training
        for column_name in self.CONTINIOUS_COLUMNS:
            self.df[column_name] = self.df[column_name].replace(r'^\s*$', np.nan, regex=True)
            self.df[column_name] = self.df[column_name].fillna(self.df[column_name].astype('float').mean())

    def __len__(self):
        return len(self.df)

    @property
    def num_categorical_cols(self):
        return len(self.CATEGORICAL_COLUMNS)

    @property
    def num_continious_cols(self):
        return len(self.CONTINIOUS_COLUMNS)

    @property
    def num_categories(self):
        """number of categories to be considered in this dataset"""
        return len(self.encoder)

    @property
    def num_classes(self):
        """number of classes in the target labels"""
        return self.df['Churn'].nunique()

    def __getitem__(self, row_index):
        continious_values = self.df.loc[row_index, self.CONTINIOUS_COLUMNS].astype('float').values.tolist()
        categorical_values_row = self.df.loc[row_index, self.CATEGORICAL_COLUMNS].values.tolist()
        categorical_values = [self.encoder.get_index(index, value) for index, value in enumerate(categorical_values_row)]
        return torch.tensor(categorical_values), torch.tensor(continious_values[::-1]), torch.tensor(continious_values[-1])
    

# ToDo: Write Unit tests
class TestBlastcharDataset(unittest.TestCase):
    CSV_PATH = '/nfs_home/janhavi2021/spring2021/tabTransformer/data/Telco-Customer-Churn.csv'

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = BlastcharDataset(cls.CSV_PATH)

    def test_length_of_dataset(self):
        df = pd.read_csv(self.CSV_PATH)
        self.assertEqual(len(df), len(self.dataset))

if __name__ == '__main__':
    unittest.main()