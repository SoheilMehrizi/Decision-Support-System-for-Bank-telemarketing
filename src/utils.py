def compare_columns_and_print_differences(df1, df2):
    """
    Compare the column names of two data frames and print the differences.

    Parameters:
        df1 (pd.DataFrame): First data frame.
        df2 (pd.DataFrame): Second data frame.

    Returns:
        list: List of column names that are different between the two data frames.
    """
    def compare_columns(df1, df2):
        columns_df1 = set(df1.columns)
        columns_df2 = set(df2.columns)
        return list(columns_df1.symmetric_difference(columns_df2))

    differences = compare_columns(df1, df2)
    print("Columns with differences:", differences)
    return differences