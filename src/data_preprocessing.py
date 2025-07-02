import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .data_cleaning import DataCleaningTransformer, BoxplotOutlierRemover
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

def cleaning_pipeline_step(
        df,
        missing_method = 'drop',
        fill_value=None,
        fill_method=None,
        missing_threshold=0.5,
        outlier_removal=False,
        ):
    
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    steps = [
        ('cleaning', DataCleaningTransformer(
            missing_method=missing_method,
            fill_value=fill_value,
            fill_method=fill_method,
            missing_threshold=missing_threshold
        )),
    ]

    if outlier_removal:
        steps.append(('outlier_removal', 
                      BoxplotOutlierRemover(numeric_features=numeric_features)))

    full_cleaning_pipeline = Pipeline(steps)

    df_cleaned = full_cleaning_pipeline.fit_transform(df)

    return df_cleaned