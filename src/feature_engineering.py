
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np



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
