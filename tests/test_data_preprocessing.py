import pandas as pd
import pytest
from src.data_preprocessing import process_dataset

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "cat1": ["a", "b", None],
        "num1": [1, 2, 3],
        "target": [1, 0, 1]
    })

def test_process_dataset_with_target_and_flags(sample_data):
    # Apply preprocessing without 'include_flags'
    df_processed = process_dataset(sample_data, "target")

    # Ensure output is a DataFrame
    assert isinstance(df_processed, pd.DataFrame)

    # Check for presence of transformed categorical/encoded features
    expected_cols = df_processed.columns.tolist()
    assert any(col.startswith("cat__cat1_") for col in expected_cols)
    assert any(col.startswith("num__num1") or col == "num1" for col in expected_cols)

    # Check that target column is retained
    assert "target" in df_processed.columns

def test_process_dataset_reset_index_and_array(sample_data):
    # Apply preprocessing
    df_processed = process_dataset(sample_data, "target")

    # Confirm it's a DataFrame and index is reset
    assert isinstance(df_processed, pd.DataFrame)
    assert df_processed.index.is_monotonic_increasing
    assert df_processed.index[0] == 0
