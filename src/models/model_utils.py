import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from tqdm import tqdm

def calculate_alift(y_true, y_proba, num_bins=20):

    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)
    df['decile'] = pd.cut(df['y_proba'], bins=num_bins, labels=False)
    lift_table = df.groupby('decile')['y_true'].mean().reset_index(name='rate')
    baseline = df['y_true'].mean()
    lift_table['lift'] = lift_table['rate'] / baseline
    max_alift = lift_table['lift'].max()
    
    return lift_table, max_alift


def plot_alift(model_name ,lift_table, max_alift, ax=None, label = 'Model Lift'):

    if ax is None:
        ax = plt.gca()
    
    lift_table = lift_table.sort_values('decile')
    ax.plot(lift_table['decile'] + 1, lift_table['lift'], marker='o', linestyle='-', color='blue', label=f"{label}(max={max_alift:.2f})")

    ax.axhline(y=1, color='gray', linestyle='--', label='Baseline (Lift = 1)')

    ax.set_title(f'{model_name}_Lift Chart')
    ax.set_xlabel('Decile')
    ax.set_ylabel('Lift')
    ax.set_xticks(range(1, lift_table['decile'].max() + 2))
    ax.legend()

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

def evaluate_model_on_test(estimator, estimator_name, X_test, y_test, ax_roc=None, ax_lift=None):
    y_test = y_test.copy().to_frame()
    y_test['y'] = y_test['y'].map({'yes': 1, 'no': 0})

    try:
        y_test_proba = estimator.predict_proba(X_test)[:, 1]
    except AttributeError:
        # For Keras models or models without predict_proba
        y_test_proba = estimator.predict(X_test).ravel()


    lift_table, max_alift = calculate_alift(y_test['y'], y_test_proba)
    test_auc_val = roc_auc_score(y_test['y'], y_test_proba)

    if (ax_lift is not None) and (ax_roc is not None):
        plot_roc_curve(y_test, y_test_proba, label=estimator_name, ax=ax_roc)
        plot_alift(model_name=estimator_name, lift_table=lift_table,max_alift=max_alift, ax=ax_lift)

    return test_auc_val, max_alift




# calculate features' importance from the trained random forest classifier .
def get_features_importance_df(RFpipeline: Pipeline, original_features_name:list):


    features_importance_dict = {key: 0 for key in original_features_name}

    preprocessor = RFpipeline.named_steps['preprocess']
    RandomForestClassifier = RFpipeline.named_steps['clf']
    encoded_feature_importance = RandomForestClassifier.feature_importances_

    encoded_feature_names = preprocessor[-1].get_feature_names_out(original_features_name)


    for name, importance in zip(encoded_feature_names, encoded_feature_importance):
        
        type, feature = name.split('__')

        if type == "num":
            features_importance_dict[feature] = round(importance, 2)
        elif type == "cat":
            cat_feature, _ =feature.split("_")
            features_importance_dict[cat_feature] += round(importance, 2)

    df_feature_importance = pd.DataFrame({
        "features": list(features_importance_dict.keys()),
        "importance": list(features_importance_dict.values())
    })

    return df_feature_importance