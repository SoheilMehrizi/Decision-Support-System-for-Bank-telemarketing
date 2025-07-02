from typing import Callable, Dict, Any, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.models.svc import train_svc
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
    "SVC":               train_svc,
    "NeuralNetwork":     train_mlp,
}

#TODO: Update the data preprocessing pipeline in order to prevent information leackage.
def train_log_compare_models(
    X_train, y_train, X_test, y_test,
    num_columns, cat_columns,
    models_name: list=None,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Train multiple classifiers

    Parameters
    ----------
    X_train, y_train
        Training features and labels.
    X_test, y_test
        Test features and labels.
    models_name
        a list of models name which want to train.

    Returns
    -------
    results : dict
        For each model name: {"model", "params", "auc_train", "auc_test", "alift"}.
    """
    #TODO: Update the data preprocessing pipeline in order to prevent information leackage.
    results: Dict[str, Any] = {}

    ax_lift=None
    ax_roc=None
    
    if models_name==None:
        models = MODEL_REGISTRY
    else:
        filterd_data = {k: MODEL_REGISTRY[k] for k in models_name if k in MODEL_REGISTRY}
        models = filterd_data
    models_num = len(models)
    if visualize:
        fig_, axes_   = plt.subplots(models_num, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes_.ravel()
    
    for idx, (name, trainer) in enumerate(tqdm(models.items(), desc="Models", unit="model")):
        logging.info(f"Training {name} ({idx+1}/{models_num})…")
        best_estimator_reference_pipeline, best_params, auc_train = trainer(X_train=X_train,
                                                                             y_train=y_train,
                                                                            num_columns=num_columns,
                                                                             cat_columns=cat_columns)

        logging.info(f"Evaluating {name} on test set…")
        
        if visualize:
            ax_lift=axes[idx]
            ax_roc=axes[idx+models_num]

        auc_test, alift = evaluate_model_on_test(
            estimator=best_estimator_reference_pipeline,
            estimator_name=name,
            X_test = X_test,
            y_test = y_test,
            ax_lift=ax_lift,
            ax_roc=ax_roc
        )

        results[name] = {
            "model":     best_estimator_reference_pipeline,
            "params":    best_params,
            "auc_train": auc_train,
            "auc_test":  auc_test,
            "alift":     alift,
        }

    logging.info("=== Test set performance summary ===")
    for name, res in results.items():
        logging.info(
            "%-20s  AUC=%.3f  ALIFT=%.3f",
            name,
            res["auc_test"],
            res["alift"]
        )
    if visualize:
        plt.show()
    return results