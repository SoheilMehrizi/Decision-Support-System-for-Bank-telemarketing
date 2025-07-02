import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import STL


class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that cleans a DataFrame by:
      - Dropping columns with more than a specified threshold of missing values.
      - Removing duplicate rows.
      - Handling missing values (drop or fill).
      - Correcting textual inconsistencies (trimming and lowercasing strings).
    """
    def __init__(self, missing_method='drop', fill_value=None, fill_method=None, missing_threshold=0.5):
        """
        Parameters:
            missing_method (str): 'drop' to remove rows with missing values or 'fill' to fill them.
            fill_value (scalar or dict): Value to use for filling missing values (if using 'fill').
            fill_method (str): Method for filling missing values (e.g., 'ffill' or 'bfill').
            missing_threshold (float): Threshold (0 to 1) to drop columns with excessive missing values.
        """
        self.missing_method = missing_method
        self.fill_value = fill_value
        self.fill_method = fill_method
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        # Identify columns with more than missing_threshold fraction of missing values.
        self.cols_to_drop_ = X.columns[X.isnull().mean() > self.missing_threshold]
        return self

    def transform(self, X, y=None):

        X_clean = X.copy()
        
        X_clean = X_clean.drop(columns=self.cols_to_drop_, errors='ignore')
        
        X_clean = X_clean.drop_duplicates()
        
        if self.missing_method == 'drop':
            X_clean = X_clean.dropna()
        elif self.missing_method == 'fill':
            if self.fill_method:
                X_clean = X_clean.fillna(method=self.fill_method)
            if self.fill_value is not None:
                X_clean = X_clean.fillna(self.fill_value)
        else:
            raise ValueError("Invalid missing_method. Use 'drop' or 'fill'.")
        
        # Correct textual inconsistencies for object-type columns.
        for col in X_clean.select_dtypes(include=['object']).columns:
            X_clean[col] = X_clean[col].str.strip().str.lower()
        
        return X_clean
    


class BoxplotOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.whisker_bounds_ = {}

    def fit(self, X, y=None):
        # Store whisker bounds using IQR method
        for feature in self.numeric_features:
            q1 = X[feature].quantile(0.05)
            q3 = X[feature].quantile(0.95)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.whisker_bounds_[feature] = (lower, upper)
        return self

    def transform(self, X):
        X_ = X.copy()
        is_outlier = np.full(X_.shape[0], False)

        for feature in self.numeric_features:
            lower, upper = self.whisker_bounds_[feature]
            is_outlier |= (X_[feature] < lower) | (X_[feature] > upper)

        # Return only non-outlier rows, and drop any temp columns
        X_clean = X_.loc[~is_outlier].copy()
        return X_clean.reset_index(drop=True)
