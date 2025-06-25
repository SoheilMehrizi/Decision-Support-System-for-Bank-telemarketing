import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .data_cleaning import DataCleaningTransformer, SeasonalDecompositionOutlierRemover
from .feature_engineering import DynamicNumToCatTransformer
import warnings

def create_preprocessor(num_features:list, cat_features:list):
    
    num_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True))
    ])

    # Categorical pipeline: one-hot encoding
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    return preprocessor


def preprocess_pipeline(
        num_columns:list,
        cat_columns:list,
        missing_method = 'drop',
        fill_value=None,
        fill_method=None,
        missing_threshold=0.5,
        outlier_removal=False,
        ):
    steps = [
        ('cleaning', DataCleaningTransformer(
            missing_method=missing_method,
            fill_value=fill_value,
            fill_method=fill_method,
            missing_threshold=missing_threshold
        )),
    ]

    if outlier_removal:
        steps.append(('outlier_removal', SeasonalDecompositionOutlierRemover(
            numeric_columns=num_columns, cat_columns=cat_columns, period=12)))

    column_transformer = create_preprocessor(num_features=num_columns, cat_features=cat_columns)

    steps.append(('column_transformer', column_transformer))

    full_preprocessing_pipeline = Pipeline(steps)

    return full_preprocessing_pipeline