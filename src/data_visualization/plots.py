import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_feature_composition(df, num_columns, cat_columns):
    """
    Plots the composition of all features in the DataFrame to look for outliers.

    Parameters:
    - df: pandas DataFrame
    - num_columns: list of numerical column names
    - cat_columns: list of categorical column names
    """
    # Combine numerical and categorical columns for unified plotting
    all_columns = num_columns + cat_columns
    total_plots = len(all_columns)
    rows = 4
    cols = math.ceil(total_plots / rows)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, col in enumerate(all_columns):
        ax = axes[i]
        if col in num_columns:
            # Plot numerical features using box plots
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Box Plot of {col}")
        elif col in cat_columns:
            # Plot categorical features using bar plots
            sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
            ax.set_title(f"Bar Plot of {col}")
            ax.tick_params(axis='x', rotation=45)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2, w_pad=2)
    plt.show()