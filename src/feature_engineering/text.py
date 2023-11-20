# Imports
import numpy as np
import pandas as pd

from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class AWBTfidfVectorizer(TransformerMixin):

    def __init__(self, use_idf=True):
        self.vectorizer = TfidfVectorizer(use_idf=True)

    def fit(self, X: Union[pd.DataFrame, np.array],
            y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:

        # Turn dataframe column to list
        X_ = [i_[0] for i_ in list(X.values.tolist())]
        X_ = self.vectorizer.fit_transform(X_)
        return X_, y

    def transform(self, X: Union[pd.DataFrame, np.array],
                  y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        X_ = [i_[0] for i_ in list(X.values.tolist())]
        X_ = self.vectorizer.transform(X_)
        return X_, y

    def fit_transform(self, X: Union[pd.DataFrame, np.array],
                      y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y)