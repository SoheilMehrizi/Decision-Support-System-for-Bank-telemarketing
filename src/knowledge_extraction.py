import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def extract_surrogate_rules(pipeline, X: pd.DataFrame, y_pred_pipeline: np.ndarray,
                            feature_names: list, max_depth: int = 3) -> pd.DataFrame:
    """
    Extract surrogate decision tree rules from a scikit-learn pipeline with preprocessing and a RandomForestClassifier.

    Args:
        pipeline (Pipeline): Fitted pipeline including preprocessing and classifier.
        X (pd.DataFrame): Original input data (before preprocessing).
        y_pred_pipeline (np.ndarray): Labels predicted by the pipeline for each X.
        feature_names (list): List of original feature names.
        max_depth (int): Maximum depth for the surrogate decision tree.

    Returns:
        pd.DataFrame: A dataframe with columns: "rule", "support", "confidence"
    """
    # Step 1: Extract preprocessing step and transform X
    preprocessor = pipeline.named_steps['preprocessor'] if 'preprocessor' in pipeline.named_steps else pipeline.steps[0][1]

    if isinstance(preprocessor, ColumnTransformer):
        transformed_X = preprocessor.transform(X)
        try:
            transformed_feature_names = preprocessor.get_feature_names_out(feature_names)
        except:
            transformed_feature_names = [f"f{i}" for i in range(transformed_X.shape[1])]
    else:
        transformed_X = preprocessor.transform(X)
        transformed_feature_names = feature_names

    # Step 2: Fit surrogate decision tree
    surrogate = DecisionTreeClassifier(max_depth=max_depth)
    surrogate.fit(transformed_X, y_pred_pipeline)

    # Step 3: Extract rules from surrogate
    tree = surrogate.tree_
    features = [
        transformed_feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]

    rules = []

    def recurse(node, rule_conditions):

        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = features[node]
            type, feature = name.split("__")

            threshold = tree.threshold[node]

            if type == "cat":
                name, value = feature.split("_")
                recurse(tree.children_left[node], rule_conditions + [f"{name} != {value}"])
                # Right
                recurse(tree.children_right[node], rule_conditions + [f"{name} == {value}"])
            else:
                # Left
                recurse(tree.children_left[node], rule_conditions + [f"{feature} <= {threshold:.3f}"])
                # Right
                recurse(tree.children_right[node], rule_conditions + [f"{feature} > {threshold:.3f}"])
        else:
            samples = tree.n_node_samples[node]
            value = tree.value[node][0]
            predicted_class = np.argmax(value)
            confidence = value[predicted_class] / np.sum(value)
            support = samples
            rule = "IF " + " AND ".join(rule_conditions) + f" THEN class={predicted_class}"
            rules.append({
                "rule": rule,
                "support": round(support, 3),
                "confidence": round(confidence, 3)
            })

    recurse(0, [])

    return pd.DataFrame(rules)




def extract_local_rules(pipeline, X: pd.DataFrame, y_pred_pipeline: np.ndarray, 
                        input_conditions: dict, feature_names: list, max_depth: int = 3) -> pd.DataFrame:
    """
    Extract local surrogate rules based on specific input feature conditions.

    Args:
        pipeline: trained sklearn pipeline with preprocessor and classifier.
        X: original dataset (before preprocessing).
        y_pred_pipeline: predictions from the pipeline.
        input_conditions: dict of known feature values (e.g., {'a': 1, 'b': 2}).
        feature_names: original feature names.
        max_depth: depth of the surrogate tree.

    Returns:
        DataFrame with human-readable rules for the local subset.
    """
    # Step 1: Filter dataset based on known feature values
    mask = pd.Series(True, index=X.index)
    for feat, val in input_conditions.items():
        mask &= (X[feat] == val)
    X_local = X[mask]
    
    if X_local.empty:
        raise ValueError("No rows match the given input conditions.")

    y_local = y_pred_pipeline[mask]

    # Step 2: Extract rules on the local dataset
    rules_df = extract_surrogate_rules(
        pipeline=pipeline,
        X=X_local,
        y_pred_pipeline=y_local,
        feature_names=feature_names,
        max_depth=max_depth
    )

    return rules_df