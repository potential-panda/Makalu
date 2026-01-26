"""
Model evaluation module for assessing LightGBM model performance.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model import LGBMTrainer

LOG = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    roc_curve: Optional[Dict[str, np.ndarray]] = None
    pr_curve: Optional[Dict[str, np.ndarray]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"metrics": self.metrics}
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        if self.classification_report is not None:
            result["classification_report"] = self.classification_report
        return result

    def summary(self) -> str:
        """Get a summary string of the evaluation results."""
        lines = ["=" * 50, "Evaluation Results", "=" * 50]

        for metric, value in self.metrics.items():
            lines.append(f"{metric}: {value:.4f}")

        if self.classification_report:
            lines.append("\n" + self.classification_report)

        if self.confusion_matrix is not None:
            lines.append("\nConfusion Matrix:")
            lines.append(str(self.confusion_matrix))

        return "\n".join(lines)


@dataclass
class Evaluator:
    """
    Evaluator for LightGBM models.

    Provides comprehensive evaluation metrics for both classification
    and regression tasks.
    """

    model: LGBMTrainer
    threshold: float = 0.5  # Classification threshold

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        return_predictions: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate the model on given data.

        Args:
            X: Features
            y: True labels
            return_predictions: Whether to include predictions in result

        Returns:
            EvaluationResult with metrics and optional predictions
        """
        if self.model.task == "classification":
            return self._evaluate_classification(X, y, return_predictions)
        else:
            return self._evaluate_regression(X, y, return_predictions)

    def _evaluate_classification(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        return_predictions: bool = False,
    ) -> EvaluationResult:
        """Evaluate classification model."""
        # Get predictions
        y_prob = self.model.predict_proba(X)
        y_pred = (y_prob >= self.threshold).astype(int)
        y_true = np.array(y)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "log_loss": log_loss(y_true, y_prob),
        }

        # ROC AUC (only if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)

        # ROC curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        else:
            roc = None

        # Precision-Recall curve
        if len(np.unique(y_true)) > 1:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            pr = {"precision": precision, "recall": recall, "thresholds": thresholds}
        else:
            pr = None

        result = EvaluationResult(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            roc_curve=roc,
            pr_curve=pr,
        )

        if return_predictions:
            result.predictions = y_pred
            result.probabilities = y_prob

        LOG.info(
            f"Classification evaluation: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )

        return result

    def _evaluate_regression(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        return_predictions: bool = False,
    ) -> EvaluationResult:
        """Evaluate regression model."""
        # Get predictions
        y_pred = self.model.predict(X)
        y_true = np.array(y)

        # Calculate metrics
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        # Mean Absolute Percentage Error (avoiding division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics["mape"] = np.nan

        result = EvaluationResult(metrics=metrics)

        if return_predictions:
            result.predictions = y_pred

        LOG.info(
            f"Regression evaluation: rmse={metrics['rmse']:.4f}, " f"mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}"
        )

        return result

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        shuffle: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.

        Uses expanding window approach suitable for time-series data:
        - Train on data up to split point
        - Test on data after split point

        Args:
            X: Features
            y: Labels
            n_splits: Number of splits
            shuffle: Whether to shuffle (not recommended for time-series)

        Returns:
            Dictionary of metric lists for each fold
        """
        from sklearn.model_selection import TimeSeriesSplit

        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        y = np.array(y) if isinstance(y, pd.Series) else y

        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            LOG.info(f"Cross-validation fold {fold + 1}/{n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and train a new model for this fold
            fold_trainer = LGBMTrainer(
                task=self.model.task,
                params=self.model.get_params(),
                num_boost_round=self.model.num_boost_round,
                early_stopping_rounds=self.model.early_stopping_rounds,
                verbose_eval=0,  # Suppress output during CV
            )

            fold_trainer.fit(X_train, y_train, X_test, y_test)

            # Evaluate
            fold_evaluator = Evaluator(fold_trainer, threshold=self.threshold)
            fold_result = fold_evaluator.evaluate(X_test, y_test)

            # Collect metrics
            for metric, value in fold_result.metrics.items():
                if metric not in results:
                    results[metric] = []
                results[metric].append(value)

        # Log summary
        LOG.info("Cross-validation results:")
        for metric, values in results.items():
            LOG.info(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

        return results

    def find_best_threshold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metric: str = "f1",
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Find the best classification threshold based on a metric.

        Args:
            X: Features
            y: True labels
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            thresholds: Array of thresholds to try. Default: 0.1 to 0.9

        Returns:
            Dictionary with best threshold and metrics at each threshold
        """
        if self.model.task != "classification":
            raise ValueError("Threshold optimization only for classification")

        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.05)

        y_prob = self.model.predict_proba(X)
        y_true = np.array(y)

        results = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            result = {
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        best_idx = results_df[metric].idxmax()
        best_threshold = results_df.loc[best_idx, "threshold"]

        LOG.info(
            f"Best threshold for {metric}: {best_threshold:.2f} " f"({metric}={results_df.loc[best_idx, metric]:.4f})"
        )

        return {
            "best_threshold": best_threshold,
            "best_metrics": results_df.loc[best_idx].to_dict(),
            "all_results": results_df,
        }

    def analyze_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Analyze predictions with detailed breakdown.

        Args:
            X: Features
            y: True labels

        Returns:
            DataFrame with predictions and analysis
        """
        if self.model.task == "classification":
            y_prob = self.model.predict_proba(X)
            y_pred = (y_prob >= self.threshold).astype(int)

            analysis = pd.DataFrame(
                {
                    "y_true": np.array(y),
                    "y_pred": y_pred,
                    "probability": y_prob,
                    "correct": np.array(y) == y_pred,
                    "confidence": np.abs(y_prob - 0.5) * 2,  # 0 to 1 scale
                }
            )

            # Add prediction category
            analysis["category"] = "TN"
            analysis.loc[(analysis["y_true"] == 1) & (analysis["y_pred"] == 1), "category"] = "TP"
            analysis.loc[(analysis["y_true"] == 0) & (analysis["y_pred"] == 1), "category"] = "FP"
            analysis.loc[(analysis["y_true"] == 1) & (analysis["y_pred"] == 0), "category"] = "FN"

        else:
            y_pred = self.model.predict(X)
            y_true = np.array(y)

            analysis = pd.DataFrame(
                {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": y_pred - y_true,
                    "abs_error": np.abs(y_pred - y_true),
                    "pct_error": np.where(y_true != 0, (y_pred - y_true) / y_true * 100, np.nan),
                }
            )

        return analysis
