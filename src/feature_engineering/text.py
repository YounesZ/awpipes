# Imports
import re
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt


from os import getcwd, sep, path
from math import ceil
from copy import deepcopy
from typing import Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, plot_confusion_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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