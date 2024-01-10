#| label: preprocessing-script
#| echo: True
#| output: False
#| filename: preprocessor.py
#| code-line-numbers: true

import os
import tarfile
import tempfile
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def preprocess(base_directory):
    """ 
    This function loads the supplied data, splits it and transforms it.
    """
    target_transfomer = ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])]
    )

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )   

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude='object')),
            ("categorical", categorical_transformer, ["island"])
        ]
    )
