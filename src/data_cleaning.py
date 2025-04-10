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
    


class SeasonalDecompositionOutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes harsh outliers from numeric columns based on seasonal decomposition.
    For each numeric column, the STL decomposition is used to obtain the residuals.
    Rows are removed if any residual is outside of [Q1 - factor*IQR, Q3 + factor*IQR],
    where factor is set high (e.g., 3.0) so that slight anomalies remain.
    
    Parameters:
        numeric_columns (list): List of numeric column names to process. If None,
                                auto-detects numeric columns.
        period (int): The seasonal period to use in STL decomposition.
        factor (float): The multiplier for IQR to define harsh outliers.
    """
    def __init__(self, numeric_columns=None, period=12, factor=3.0):
        self.numeric_columns = numeric_columns
        self.period = period
        self.factor = factor

    def fit(self, X, y=None):
        # Determine numeric columns if not provided.
        if self.numeric_columns is None:
            self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_columns_ = self.numeric_columns

        # Compute bounds for each column based on the residuals from STL.
        self.bounds_ = {}
        for col in self.numeric_columns_:
            series = X[col]
            # Apply STL decomposition; robust=True helps mitigate the effect of extreme outliers
            stl = STL(series, period=self.period, robust=True)
            resid = stl.fit().resid
            # Calculate IQR-based thresholds for residuals
            Q1 = np.percentile(resid, 5)
            Q3 = np.percentile(resid, 95)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        mask = np.ones(len(X_out), dtype=bool)
        # For each numeric column, recompute the residuals and apply the pre-computed thresholds
        for col in self.numeric_columns_:
            series = X_out[col]
            stl = STL(series, period=self.period, robust=True)
            resid = stl.fit().resid
            lower_bound, upper_bound = self.bounds_[col]
            col_mask = (resid >= lower_bound) & (resid <= upper_bound)
            mask &= col_mask
        # Return only the rows where all numeric residuals are within acceptable bounds
        return X_out[mask]