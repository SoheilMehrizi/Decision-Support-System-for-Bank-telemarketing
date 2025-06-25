import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve, GridSearchCV, ParameterGrid

from tqdm import tqdm

def calculate_alift(y_true, y_proba, num_bins=10):

    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)
    df['decile'] = pd.qcut(df['y_proba'], q=num_bins, labels=False)
    lift_table = df.groupby('decile')['y_true'].mean().reset_index(name='rate')
    baseline = df['y_true'].mean()

    lift_table['lift'] = lift_table['rate'] / baseline
    alift = lift_table['lift'].mean()
    return lift_table, alift


def plot_alift(model_name ,lift_table, ax=None, label = 'Model Lift'):

    if ax is None:
        ax = plt.gca()
    
    lift_table = lift_table.sort_values('decile')
    ax.plot(lift_table['decile'] + 1, lift_table['lift'], marker='o', linestyle='-', color='blue', label=label)

    ax.axhline(y=1, color='gray', linestyle='--', label='Baseline (Lift = 1)')

    ax.set_title(f'{model_name}_Lift Chart')
    ax.set_xlabel('Decile')
    ax.set_ylabel('Lift')
    ax.set_xticks(range(1, lift_table['decile'].max() + 2))
    ax.legend()
    ax.grid(True)

def plot_roc_curve(y_true, y_proba, label=None, ax=None):
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        ax = plt.gca()
    ax.set_title(f'{label}_ROC_AUC')

    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    return auc_val

# TODO: Refactor, to make it more dynamic.

def plot_learning_curve(estimator, X, y, cv, scoring='roc_auc'):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def evaluate_model_on_test(estimator, estimator_name, X_test, y_test, ax_roc=None, ax_lift=None):
    y_test = y_test.copy()
    y_test = y_test.to_frame()
    
    y_test['y'] = y_test['y'].map({'yes': 1, 'no': 0})
    
    if estimator_name == "NeuralNetwork":
        y_test_proba = estimator.predict(X_test)[:, 0]
        print("y_test_proba:",y_test_proba.shape)
    else:
        y_test_proba = estimator.predict_proba(X_test)[:, 1]
    lift_table, alift = calculate_alift(y_test['y'], y_test_proba)

    if (ax_lift is not None) and (ax_roc is not None):
        test_auc_val = plot_roc_curve(y_test, y_test_proba, label=estimator_name, ax=ax_roc)
        plot_alift(model_name = estimator_name, lift_table = lift_table, ax = ax_lift)

    return test_auc_val, alift