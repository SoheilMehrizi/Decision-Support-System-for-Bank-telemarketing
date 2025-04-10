
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# fe : feature engineering 
# def convert_numeric_to_cat_fe(df, config:dict, drop_raw=True):
#     """
#     there are a set of numerical features that are more promising to be treated as categorical features .
#     This module converts less distributed numerical features into categorical features and removes the raw features.
    
#     Parameters:
#         1. df: (pandas.DataFrame) the data frame that some of which features should be transformed .
#         2. config:(dict) a dictionary that tells the function what features need transformation and in what settign.
#         3. drop_raw (boolean): tells the function to drop original features or not .
        
#     After creating these categorical features, the transformer removes the original
#     numerical features if drop_raw been set to True.
#     """

#     df_transformed = df.copy()

#     for col, params in config.items():
#         new_col = params['new_name']
#         bins = params['bins']
#         labels = params['labels']

#         if 'special' in params and 'special_label' in params:
#             df_transformed[new_col] = np.where(
#                 df_transformed[col] == params['special'],
#                 params['special_label'],
#                 pd.cut(df_transformed[col], bins=bins, labels=labels)
#             )
#         else:
#             df_transformed[new_col] = pd.cut(df_transformed[col], bins=bins, labels=labels)
    
#     if drop_raw:
#         raw_cols = list(config.keys())
#         df_transformed.drop(columns=raw_cols, inplace=True)

#     return df_transformed



# Converting under-distributed numerical features into categorical ones using a dynamic, parameter-driven transformer.
class DynamicNumToCatTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that converts selected numeric features into categorical features based on configuration.
    
    Parameters:
      config (dict): Dictionary with configuration for each feature.
      drop_raw (bool): If True, drop the raw numeric column after conversion.
    """
    def __init__(self, config, drop_raw=True):
        self.config = config
        self.drop_raw = drop_raw

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for col, params in self.config.items():
            new_col = params['new_name']
            bins = params['bins']
            labels = params['labels']
            if 'special' in params and 'special_label' in params:
                df[new_col] = np.where(
                    df[col] == params['special'],
                    params['special_label'],
                    pd.cut(df[col], bins=bins, labels=labels)
                )
            else:
                df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
        if self.drop_raw:
            df.drop(columns=list(self.config.keys()))
        return df
