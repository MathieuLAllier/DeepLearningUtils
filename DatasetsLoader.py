import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


class StructuredDataset(Dataset):
    def __init__(self, X, Y, categorical=None):
        self.X = X
        self.Y = Y
        self.categorical = categorical

    def __getitem__(self, idx):
        return [
            np.asarray(self.X[self.categorical].iloc[idx]).astype(int),
            np.asarray(self.X.drop(columns=self.categorical).iloc[idx]).astype(float),
            np.asarray(self.Y.iloc[idx]).astype(int)
            ]

    def __len__(self):
        return len(self.df)

    def __getattr__(self, item):
        attributes = {
            'n_cols': len(self.X.columns),
            'n_cat': len(self.X[self.categorical].columns),
            'n_cont': len(self.X.drop(columns=self.categorical).columns)
        }

        return attributes.get(item, attributes)

    @classmethod
    def from_df(cls, df, y):

        y = y if isinstance(y, list) else [y]
        categories = df.select_dtypes(include='category')

        return cls(X=df.drop(columns=y), Y=df[y], categorical=categories)
