import re
import numpy as np
import pandas as pd

from os import getcwd, sep, path
from copy import deepcopy
from typing import Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder


class AWBLabelEncoder(TransformerMixin):

    def __init__(self, test_size=0.2, shuffle=True, random_state=42):
        self.is_fit = False
        self.encoder= LabelEncoder()
        TransformerMixin.__init__(self)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)
        return X, y_encoded

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # Filter labels
        filter_out  = np.ones(y.shape).astype(bool)
        for i_ in y.drop_duplicates().values.flatten():
            try:
                self.encoder.transform([i_])
            except:
                filter_out[ y==i_] = False
                print(f'Unknown class {i_}')

        # Make filter
        X_ = X[filter_out]
        y_ = self.encoder.transform(y[filter_out])
        return X_, y_

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, *kwargs)


class AWBShapeAs2D(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X: np.array, y: np.array, **kwargs) -> np.array:
        y_ = y
        if len(np.shape(y)) == 1:
            y_ = np.reshape(y, [-1, 1])

        return X, y_

    def transform(self, X: np.array, y: np.array, **kwargs) -> np.array:
        return self.fit(X, y, **kwargs)

    def fit_transform(self, X: np.array, y: np.array, **kwargs) -> np.array:
        return self.fit(X, y, **kwargs)