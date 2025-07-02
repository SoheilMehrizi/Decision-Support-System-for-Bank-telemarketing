import numpy as np
import pandas as pd
import tempfile

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline

import tensorflow as tf
from keras import layers, callbacks, optimizers, metrics
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import keras_tuner as kt

from configs.config_repository import ConfigRepository
from src.models.model_repository import ModelRepository
from sklearn.preprocessing import FunctionTransformer

from src.data_preprocessing import create_preprocessor

def build_model(hp,input_dim: int, mlp_config: dict) -> tf.keras.Model:
    """
    Build a Keras Sequential model using hyperparameters from configuration.
    """
    model = tf.keras.Sequential()

    num_configs = len(mlp_config['hidden_layers'])
    idx = hp.Choice('hidden_layers_idx', values=list(range(num_configs)))
    hidden_layers = mlp_config['hidden_layers'][idx]

    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', input_shape=(input_dim,)))
        dropout_rate = hp.Choice('dropout_rate', values=mlp_config['dropout_rate'])
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    learning_rate = hp.Choice('learning_rate', values=mlp_config['learning_rate'])
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[metrics.AUC(name='auc')]
    )
    return model


def train_mlp(X_train, y_train,
              num_columns, cat_columns,
              dev_size:int = 0.2,
              random_state: int = 42) -> tuple:
    """
        Train a Multilayer Perceptron (MLP) for binary classification with hyperparameter tuning,
        proper preprocessing, and class balancing. Returns a fitted pipeline containing preprocessing 
        and the trained model.

        Steps performed:
        - Preprocessing with column-wise transformation
        - Class balancing via RandomOverSampler
        - Hyperparameter search using Keras Tuner (Hyperband)
        - Final retraining on the full dataset with best parameters
        - Model and metadata logging via ModelRepository

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature set (raw format, before preprocessing).
        y_train : pd.Series or pd.DataFrame
            Training labels. Must be binary with values 'yes' and 'no'.
        num_columns : list of str
            List of numerical feature names.
        cat_columns : list of str
            List of categorical feature names.
        dev_size : float, optional (default=0.2)
            Proportion of the data to use as validation during hyperparameter tuning.
        random_state : int, optional (default=42)
            Random seed for reproducibility.

        Returns
        -------
        tuple
            A 3-tuple:
            - reference_pipeline : sklearn.pipeline.Pipeline
                The final fitted pipeline containing the preprocessing step and trained model.
            - best_hps : keras_tuner.HyperParameters
                The best hyperparameter configuration found during tuning.
            - auc : float
                ROC AUC score on the validation set using the best model.
    """

    repo_cfg = ConfigRepository(config_path="../configs/models_config.json")
    mlp_config = repo_cfg.get_config('mlp')

    if isinstance(y_train, pd.Series):
        y = y_train.map({'yes': 1, 'no': 0})
    else:
        y = y_train['y'].map({'yes': 1, 'no': 0})
    
    X = X_train

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=dev_size,
        random_state=random_state,
        shuffle=True,
        stratify=y
    )
    preprocessing_pipeline = create_preprocessor(num_features=num_columns, 
                                                 cat_features=cat_columns)
    
    preprocessing_pipeline.fit(X_train, y_train)
    X_tr_prp = preprocessing_pipeline.transform(X_train)
    X_val_prp = preprocessing_pipeline.transform(X_val)

    

    ros = RandomOverSampler(random_state=random_state)
    X_tr_rsmpld, y_tr_rsmpld = ros.fit_resample(X_tr_prp, y_train)

    stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=mlp_config.get('patience', 10),
        mode='max', restore_best_weights=True
    )

    input_dim = X_tr_rsmpld.shape[1]
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, input_dim, mlp_config),
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=mlp_config.get('max_epochs', 50),
        factor=mlp_config.get('factor', 3),
        directory=tempfile.mkdtemp(),
        project_name=f"multi layer preceptron_tuner"
    )


    #TODO: Update the data preprocessing pipeline in order to prevent information leackage.
    tuner.search(
        X_tr_rsmpld, y_tr_rsmpld,
        epochs= mlp_config.get('search_epochs', 50),
        validation_data=(X_val_prp, y_val),
        callbacks=[stop],
        verbose=3
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    y_val_pred = best_model.predict(X_val_prp).ravel()

    auc = roc_auc_score(y_val, y_val_pred)

    
    best_auc = auc
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"\nBest Candidate's AUC: {auc:.4f}")
    print("Best hyperparameters:", best_hps.values)
    best_model.summary()
    
    print(5*"*","Retrain the best Estimator on the full training set",5*"*")
    # FINAL TRAINING
    X_full_prp = preprocessing_pipeline.fit_transform(X)    
    X_full_rsmpld, y_full_rsmpld = ros.fit_resample(X_full_prp, y)

    input_dim = X_full_rsmpld.shape[1]

    final_model = build_model(best_hps, input_dim, mlp_config)
    final_model.fit(
        X_full_rsmpld, y_full_rsmpld,
        epochs=mlp_config.get('final_epochs', 50),
        batch_size=mlp_config.get('batch_size', 32),
        verbose=2
    )

    refernce_pipeline = ImbPipeline([
       ('preprocess', preprocessing_pipeline),
       ("clf", final_model)
    ])

    repo_mdl = ModelRepository(experiment_name=mlp_config.get('experiment_name', 'MLP_Models'))
    run_id = repo_mdl.log_model(
        refernce_pipeline,
        model_name=mlp_config.get('model_name', 'MLP'),
        params=best_hps.values,
        metrics={'roc_auc': best_auc}
    )
    registered = repo_mdl.register_model(
        run_id,
        model_name=mlp_config.get('model_name', 'MLP'),
        registered_name=mlp_config.get('registered_name', 'MLP')
    )

    return refernce_pipeline, best_hps, auc
