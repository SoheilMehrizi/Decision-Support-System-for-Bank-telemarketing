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
    # Numerical pipeline: normalization (mean=0, std=1)
    
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


class DataFrameOutputTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_transformer, index=None):
        self.col_transformer = col_transformer
        self.index = index

    def fit(self, X, y=None):
        try:
            # Get feature names from the ColumnTransformer
            self.feature_names_ = self.col_transformer.get_feature_names_out()
        except AttributeError:
            # Fallback to generic feature names if unavailable
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X, y=None):
        # Transform the data
        transformed = X
        # Dynamically adjust feature names if there is a mismatch
        if len(self.feature_names_) != transformed.shape[1]:
            self.feature_names_ = [f"feature_{i}" for i in range(transformed.shape[1])]
        # Return a DataFrame with the correct index and column names
        if self.index is not None and len(self.index) == transformed.shape[0]:
            return pd.DataFrame(transformed, columns=self.feature_names_, index=self.index)
        else:
            return pd.DataFrame(transformed, columns=self.feature_names_)



def process_dataset(
    df_,
    target_column,
    missing_method='drop',
    fill_value=None,
    fill_method=None,
    missing_threshold=0.5,
    outlier_removal=False,
    add_new_features = False,
    fe_config = {},
    outlier_factor=3,
    return_dataframe=True
):
    df_ = df_.copy()

    if target_column and target_column in df_.columns.tolist():
        df_[target_column] = df_[target_column].map({'yes': 1, 'no': 0})
        y = df_[target_column]
        X = df_.drop(columns=[target_column])
    # else:
        X = df_.copy()
        y = None

    num_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()



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
            numeric_columns=num_columns, period=12, factor=outlier_factor
        )))
    if add_new_features and fe_config:
        
        for feature, config in fe_config.items():
            new_feature_name = config.get('new_feature_name', f"{feature}_category")
            cat_columns.append(new_feature_name)

        steps.append(("feature_engineering", DynamicNumToCatTransformer(fe_config, drop_raw=False)))
    
    preprocessor = create_preprocessor(num_features=num_columns, cat_features=cat_columns)
    steps.append(('preprocessing', preprocessor))

    if return_dataframe:
        steps.append(('to_dataframe', DataFrameOutputTransformer(col_transformer=preprocessor, index=df_.index)))

    full_pipeline = Pipeline(steps)

    transformed_features = full_pipeline.fit_transform(X)

    if y is not None:
        y_aligned = y.loc[transformed_features.index]
        transformed_features[target_column] = y_aligned

    transformed_features = transformed_features.reset_index(drop=True)
    
    if fe_config:

        columns_to_drop=[f'num__{x}' for x in fe_config.keys()]

        transformed_features = transformed_features.drop(columns = columns_to_drop)
    return transformed_features