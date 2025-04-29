import pandas as pd
import pytest
from src.data_preprocessing import process_dataset

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "cat1": ["a", "b", None],
        "num1": [1, 2, 3],
        "balance": [100, 200, 300],
        "y": ["yes", "no", "yes"]
    })


def test_process_dataset_with_target_and_flags(sample_data):
    # Apply preprocessing specifying 'y' as the target column
    df_processed = process_dataset(sample_data, "y")

    # Ensure output is a DataFrame
    assert isinstance(df_processed, pd.DataFrame)

    cols = df_processed.columns.tolist()

    assert "cat__cat1_a" in cols
    assert "cat__cat1_b" in cols

    assert "num__num1" in cols

    assert "y" in df_processed.columns


def test_process_dataset_reset_index_and_length(sample_data):
    df_processed = process_dataset(sample_data, "y")

    assert isinstance(df_processed, pd.DataFrame)
    assert df_processed.index.is_monotonic_increasing
    assert df_processed.index[0] == 0

    assert len(df_processed) == 2
