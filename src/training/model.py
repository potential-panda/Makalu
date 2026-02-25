"""
LightGBM model training and persistence module.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


# Default parameters for different task types
DEFAULT_PARAMS_CLASSIFICATION = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

DEFAULT_PARAMS_REGRESSION = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


@dataclass
class LGBMTrainer:
    """
    LightGBM trainer for classification and regression tasks.

    Provides a simple API for training, saving, and loading LightGBM models.
    """

    task: str = "classification"  # 'classification' or 'regression'
    params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 1000
    early_stopping_rounds: Optional[int] = 50
    verbose_eval: int = 100

    # Trained model and metadata
    model: Optional[lgb.Booster] = None
    feature_names: Optional[List[str]] = None
    best_iteration: Optional[int] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default parameters based on task type."""
        if not self.params:
            if self.task == "classification":
                self.params = DEFAULT_PARAMS_CLASSIFICATION.copy()
            else:
                self.params = DEFAULT_PARAMS_REGRESSION.copy()

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        categorical_features: Optional[List[str]] = None,
        feval: Optional[Any] = None,
    ) -> "LGBMTrainer":
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels for early stopping
            categorical_features: List of categorical feature names
            feval: Optional custom evaluation function

        Returns:
            Self for method chaining
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        LOG.info(f"Training LightGBM {self.task} model with {len(self.feature_names)} features")

        # Handle class weights if provided in params
        sample_weight = None
        if self.task == "classification" and "class_weight" in self.params:
            class_weight = self.params.pop("class_weight")
            if class_weight == "balanced":
                # Calculate balanced weights
                classes = np.unique(y_train)
                weight_dict = {}
                n_samples = len(y_train)
                for cls in classes:
                    weight_dict[cls] = n_samples / (len(classes) * np.sum(y_train == cls))
                print(f"Calculated balanced class weights: {weight_dict}")
                LOG.info(f"Calculated balanced class weights: {weight_dict}")
            elif isinstance(class_weight, dict):
                weight_dict = class_weight
                LOG.info(f"Using manual class weights: {weight_dict}")
            else:
                raise ValueError("class_weight must be 'balanced' or a dictionary {class_label: weight}")

            # Map class weights to every sample
            sample_weight = pd.Series(y_train).map(weight_dict).values

        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            feature_name=self.feature_names,
            categorical_feature=categorical_features,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                categorical_feature=categorical_features,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Training callbacks
        callbacks = []
        if self.early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds))
        if self.verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=self.verbose_eval))

        # Record training history via callback
        evals_result = {}
        callbacks.append(lgb.record_evaluation(evals_result))

        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            feval=feval,
        )

        # Store training history
        self.training_history = evals_result
        self.best_iteration = self.model.best_iteration

        LOG.info(f"Training completed. Best iteration: {self.best_iteration}")

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features to predict on
            num_iteration: Number of iterations to use. If None, uses best_iteration.

        Returns:
            Predicted values (probabilities for classification, values for regression)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if num_iteration is None:
            num_iteration = self.best_iteration

        return self.model.predict(X, num_iteration=num_iteration)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get probability predictions for classification.

        Args:
            X: Features to predict on
            num_iteration: Number of iterations to use

        Returns:
            Predicted probabilities
        """
        return self.predict(X, num_iteration)

    def predict_class(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get class predictions for binary classification.

        Args:
            X: Features to predict on
            threshold: Classification threshold
            num_iteration: Number of iterations to use

        Returns:
            Predicted classes (0 or 1)
        """
        proba = self.predict_proba(X, num_iteration)
        return (proba >= threshold).astype(int)

    def feature_importance(
        self,
        importance_type: str = "gain",
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance - 'gain', 'split', or 'shap'
            as_dataframe: Whether to return as DataFrame

        Returns:
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importance = self.model.feature_importance(importance_type=importance_type)

        if as_dataframe:
            df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importance,
                }
            )
            return df.sort_values("importance", ascending=False).reset_index(drop=True)

        return importance

    def save(self, path: str, include_metadata: bool = True) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model (without extension)
            include_metadata: Whether to save metadata alongside the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_path = path.with_suffix(".lgb")
        self.model.save_model(str(model_path))
        LOG.info(f"Saved LightGBM model to {model_path}")

        # Save full trainer state with pickle
        pkl_path = path.with_suffix(".pkl")
        state = {
            "task": self.task,
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "training_history": self.training_history,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
        LOG.info(f"Saved trainer state to {pkl_path}")

        # Save metadata as JSON for easy inspection
        if include_metadata:
            meta_path = path.with_suffix(".json")
            metadata = {
                "task": self.task,
                "params": self.params,
                "num_boost_round": self.num_boost_round,
                "best_iteration": self.best_iteration,
                "num_features": len(self.feature_names) if self.feature_names else 0,
                "feature_names": self.feature_names,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            LOG.info(f"Saved metadata to {meta_path}")

    @classmethod
    def load(cls, path: str) -> "LGBMTrainer":
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model (without extension)

        Returns:
            LGBMTrainer instance with loaded model
        """
        path = Path(path)

        # Load trainer state
        pkl_path = path.with_suffix(".pkl")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Trainer state file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        # Create trainer instance
        trainer = cls(
            task=state["task"],
            params=state["params"],
            num_boost_round=state["num_boost_round"],
            early_stopping_rounds=state["early_stopping_rounds"],
        )
        trainer.feature_names = state["feature_names"]
        trainer.best_iteration = state["best_iteration"]
        trainer.training_history = state["training_history"]

        # Load LightGBM model
        model_path = path.with_suffix(".lgb")
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM model file not found: {model_path}")

        trainer.model = lgb.Booster(model_file=str(model_path))
        LOG.info(f"Loaded model from {model_path}")

        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> "LGBMTrainer":
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        self.params.update(params)
        return self
