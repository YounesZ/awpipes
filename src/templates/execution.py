# Imports
import re
import sys
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from os import getcwd, sep, path
from copy import deepcopy
from nltk import word_tokenize

import inspect
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Modelling
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer



def standard_obj_type(obj):

    # Determine object type
    if TransformerMixin in obj.__class__.__mro__:
        out = 'transformer'

    elif (ClassifierMixin in obj.__class__.__mro__) or (RegressorMixin in obj.__class__.__mro__):
        out = 'estimator'

    elif BaseEstimator in obj.__class__.__mro__:
        out = 'transformer'

    else:
        raise ValueError(f'Pipeline step {obj} is neither a transformer nor a model')

    return out


def standard_block_execution(obj, X, y, do_fit, **kwargs):
    # Determine block outputs
    y_ = deepcopy(y)
    X_ = deepcopy(X)

    obj_t = standard_obj_type(obj)

    if obj_t == 'transformer':  # Always output X
        # Determine N_OUT
        n_out = get_transformer_outputs(obj)
        if n_out not in [1, 2]:
            raise (ValueError, f'Incorrect number of outputs from transformer {obj.__class__}')

        if do_fit:
            if n_out == 1:
                X_ = obj.fit_transform(X_, **kwargs)
            elif n_out == 2:
                X_, y_ = obj.fit_transform(X_, y_, **kwargs)
        else:
            if n_out == 1:
                X_ = obj.transform(X_)
            elif n_out == 2:
                X_, y_ = obj.transform(X_, y_, **kwargs)

    elif obj_t == 'estimator':
        n_out = 1  # Estimators only ouput y
        if do_fit:
            obj.fit(X, y, **kwargs)
        X_, y_ = y_, obj.predict(X_)

    return X_, y_
