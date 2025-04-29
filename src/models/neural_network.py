import numpy as np
import pandas as pd
import tempfile

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers, callbacks, optimizers, metrics
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import keras_tuner as kt

from configs.config_repository import ConfigRepository
from src.models.model_repository import ModelRepository

def build_model(hp, input_dim: int, mlp_config: dict) -> tf.keras.Model:
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
              test_size:int = 0.2,
              random_state: int = 42) -> tuple:
    """
    Train an MLP with hyperparameter tuning using configs and log the final model.
    """

    repo_cfg = ConfigRepository(config_path="../configs/models_config.json")
    mlp_config = repo_cfg.get_config('mlp')

    X = X_train.values
    y = y_train.values
    input_dim = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y
    )

    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=mlp_config.get('patience', 10),
        mode='max', restore_best_weights=True
    )

    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, input_dim, mlp_config),
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=mlp_config.get('max_epochs', 50),
        factor=mlp_config.get('factor', 3),
        directory=tempfile.mkdtemp(),
        project_name=f"multi layer preceptron_tuner"
    )



    tuner.search(
        X_res, y_res,
        epochs= mlp_config.get('search_epochs', 50),
        validation_data=(X_val, y_val),
        callbacks=[stop],
        verbose=3
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    y_val_pred = best_model.predict(X_val).ravel()

    auc = roc_auc_score(y_val, y_val_pred)

    
    best_auc = auc
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"\nBest Candidate's AUC: {auc:.4f}")

    final_model = build_model(best_hps, input_dim, mlp_config)
    final_model.fit(
        X, y,
        epochs=mlp_config.get('final_epochs', 50),
        batch_size=mlp_config.get('batch_size', 32),
        verbose=2
    )

    repo_mdl = ModelRepository(experiment_name=mlp_config.get('experiment_name', 'MLP_Models'))
    run_id = repo_mdl.log_model(
        final_model,
        model_name=mlp_config.get('model_name', 'MLP'),
        params=best_hps.values,
        metrics={'roc_auc': best_auc}
    )
    registered = repo_mdl.register_model(
        run_id,
        model_name=mlp_config.get('model_name', 'MLP'),
        registered_name=mlp_config.get('registered_name', 'MLP')
    )

    return final_model, best_hps, auc
