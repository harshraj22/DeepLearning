import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from pprint import pprint

sys.path.append('..')

from utils import CategoryEncoder



class BlastcharDataset(Dataset):
    """ Class for efficiently loading the Blastchar dataset.
    src: https://www.kaggle.com/blastchar/telco-customer-churn
    """

    CATEGORICAL_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    CONTINIOUS_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def __init__(self, dataset_path):
        super(BlastcharDataset, self).__init__()

        self.df = pd.read_csv(dataset_path)
        self.categoricals = self.df.loc[:,self.CATEGORICAL_COLUMNS]
        self.encoder = CategoryEncoder([self.categoricals[header].values.tolist() for header in self.CATEGORICAL_COLUMNS])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        continious_values = self.df.loc[index, self.CONTINIOUS_COLUMNS].astype('float').values.tolist()
        categorical_values_row = self.df.loc[index, self.CATEGORICAL_COLUMNS].values.tolist()
        categorical_values = [self.encoder.get_index(index, value) for index, value in enumerate(categorical_values_row)]
        return categorical_values, continious_values
    


if __name__ == '__main__':
    csv_path = '/home/prabhu/spring2021/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    _d = BlastcharDataset(csv_path)

    for i in range(5, 10):
        print(_d[i])
    # print(len(_d.encoder))
