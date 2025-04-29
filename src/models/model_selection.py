from typing import Callable, Dict, Any, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.models.svm import train_svm
from src.models.neural_network import train_mlp
from src.models.model_utils import evaluate_model_on_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ModelFactory = Callable[..., Tuple[Any, Dict[str, Any], float]]

MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "LogisticRegression": train_logistic_regression,
    "RandomForest":      train_random_forest,
    "SVC":               train_svm,
    "NeuralNetwork":     train_mlp,
}


def train_compare_models(
    X_train, y_train, X_test, y_test,
    models: Dict[str, ModelFactory] = MODEL_REGISTRY,
    n_rows: int = 2, n_cols: int = 2,
) -> Dict[str, Any]:
    """
    Train multiple classifiers, plot ROC & lift charts, and return performance metrics.

    Parameters
    ----------
    X_train, y_train
        Training features and labels.
    X_test, y_test
        Test features and labels.
    models
        Mapping from model name to a trainer function that returns (estimator, best_params, auc_train).
    n_rows, n_cols
        Grid layout for subplots.

    Returns
    -------
    results : dict
        For each model name: {"model", "params", "auc_train", "auc_test", "alift"}.
    """
    results: Dict[str, Any] = {}

    fig_roc, axes_roc   = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    fig_lift, axes_lift = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axes_roc   = axes_roc.ravel()
    axes_lift  = axes_lift.ravel()

    for idx, (name, trainer) in enumerate(tqdm(models.items(), desc="Models", unit="model")):
        logging.info(f"Training {name} ({idx+1}/{len(models)})…")
        model, best_params, auc_train = trainer(X_train, y_train)

        logging.info(f"Evaluating {name} on test set…")
        auc_test, alift = evaluate_model_on_test(
            model,
            name,
            X_test,
            y_test,
            ax_roc=axes_roc[idx],
            ax_lift=axes_lift[idx]
        )

        results[name] = {
            "model":     model,
            "params":    best_params,
            "auc_train": auc_train,
            "auc_test":  auc_test,
            "alift":     alift,
        }

    
    for fig in (fig_roc, fig_lift):
        fig.tight_layout()
    plt.show()

    logging.info("=== Test set performance summary ===")
    for name, res in results.items():
        logging.info(
            "%-20s  AUC=%.3f  ALIFT=%.3f",
            name,
            res["auc_test"],
            res["alift"]
        )

    return results