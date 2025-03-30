import pandas as pd

# Load the data 
def load_csv_to_dataframe(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, delimiter=";")
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an error parsing the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# data_cleaning.py

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    return df_cleaned

def handle_missing_values(df, method='drop', fill_value=None, fill_method=None):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        method (str): Strategy for handling missing values.
                      Options:
                        - 'drop': Remove rows with missing values.
                        - 'fill': Fill missing values.
        fill_value (scalar or dict): Value to use for filling missing values if method is 'fill'.
        fill_method (str): Method for filling missing values (e.g., 'ffill' or 'bfill').
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if method == 'drop':
        df_cleaned = df.dropna().reset_index(drop=True)
    elif method == 'fill':
        # First try method-based filling if specified.
        if fill_method:
            df_cleaned = df.fillna(method=fill_method)
        # Then, fill any remaining NaNs with a constant value if provided.
        if fill_value is not None:
            df_cleaned = df_cleaned.fillna(fill_value)
    else:
        raise ValueError("Invalid method. Use 'drop' or 'fill'.")
    
    return df_cleaned

def correct_inconsistencies(df):
    """
    Correct inconsistencies in the DataFrame.
    
    This function provides an example correction by:
      - Stripping leading/trailing whitespace
      - Converting text to lowercase for object-type columns
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with corrected inconsistencies.
    """
    # Apply string corrections on object (text) columns.
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

def clean_data(df, missing_method='drop', fill_value=None, fill_method=None):
    """
    Run the full data cleaning process:
      1. Remove duplicate rows.
      2. Handle missing values.
      3. Correct textual inconsistencies.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missing_method (str): Strategy for missing values: 'drop' or 'fill'.
        fill_value (scalar or dict): Value for filling missing values if using 'fill'.
        fill_method (str): Method for filling missing values if using 'fill' (e.g., 'ffill').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_cleaned = remove_duplicates(df)
    df_cleaned = handle_missing_values(df_cleaned, method=missing_method, fill_value=fill_value, fill_method=fill_method)
    df_cleaned = correct_inconsistencies(df_cleaned)
    return df_cleaned

# 
