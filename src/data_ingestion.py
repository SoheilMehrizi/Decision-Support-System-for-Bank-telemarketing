import pandas as pd
from deployment.app.repositories.bank_data_repository import BankDataRepository
from deployment.app.database import get_db


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


def load_data_from_db(
    db = get_db
):

    repo = BankDataRepository(db)
    data_train = repo.get_bank_data_train()
    data_test = repo.get_bank_data_test()

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    df_train = df_train.drop(columns=["training_data"])
    df_test = df_test.drop(columns=["training_data"])

    return (df_train, df_test)