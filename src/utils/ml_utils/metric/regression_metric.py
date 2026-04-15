from src.entity.artifact_entity import RegressionMetricArtifact
from src.exception.exception import CustomException
from src.logging.logger import get_logger
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.entity.config_entity import ModelTrainerConfig,TrainingPipelineConfig

import sys, os
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from src.constants.training_pipeline import SCHEMA_FILE_PATH

logging = get_logger(__name__)


training_pipeline_config=TrainingPipelineConfig()
model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
data_schema = read_yaml_file(file_path=SCHEMA_FILE_PATH)

# ── Metric ─────────────────────────────────────────────────────────────────────

def get_regression_score(y_true:pd.Series, y_pred:pd.Series)->RegressionMetricArtifact:
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        regression_metric_artifact = RegressionMetricArtifact(
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            r2_score=r2
        )
        return regression_metric_artifact
    except Exception as e:
        raise CustomException(e, sys) from e
    

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str,
) -> None:
    """Scatter plot of actual vs predicted ratings."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="steelblue", label="Predictions")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_actual_vs_pred.png")
    fig.savefig(path, dpi=120)
    plt.close()
    logging.info(f"Plot saved -> {path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str,
) -> None:
    """Residual distribution plot."""
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=12, color="coral")
    axes[0].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted Rating")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_residuals.png")
    fig.savefig(path, dpi=120)
    plt.close()
    logging.info(f"Plot saved -> {path}")


def plot_feature_importance(
    model: Any,
    feature_names: list,
    model_name: str,
    save_dir: str,
    top_n: int = 10,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        logging.warning(f"{model_name} has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close()
    logging.info(f"Plot saved -> {path}")


def compare_models(reports: Dict[str, Dict], save_dir: str) -> None:
    """Bar chart comparing all models on MAE / RMSE / R²."""
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(reports.keys())
    metrics_list = ["model_test_mae", "model_test_rmse", "model_test_r2_score"]
    data = {metric: [reports[name][metric] for name in model_names] for metric in metrics_list}

    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, vals) in enumerate(data.items()):
        ax.bar(x + i * width, vals, width, label=metric.upper())

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_title("Model Comparison (Test Set)")
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close()
    logging.info(f"Model comparison plot saved -> {path}")


def evaluate_models(X_train: np.array, y_train: np.array,
                    X_test: np.array, y_test: np.array, 
                    models: Dict, params: Dict) -> dict:
    try:
        logging.info("="*60)
        logging.info("Initiate model evaluation...")
        logging.info(f"Considered models:\n{models}")
        logging.info(f"Considered model parameters for hyperparameter tuning:\n{params}")
        logging.info("Performing hyperparameter tuning with the help of GridSearchCV" \
        "with default K-Fold Cross validation assigning K=3")

        report: Dict = {}
        models_performance: Dict = {}
        plot_save_dir: str = model_trainer_config.model_evaluation_dir

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]

            # Initializing GridSearchCV
            gs = GridSearchCV(estimator=model,
                            param_grid=param,
                            cv=3)
            # Hyperparameter tuning
            gs.fit(X_train,y_train)

            # Setting best parameter to the related model from grid search cv
            model.set_params(**gs.best_params_)
            #Training model with best parameters
            model.fit(X_train,y_train)

            # Model prediction for known values i.e. Training Dataset
            y_train_pred = model.predict(X_train)
            # Model prediction for unknown values i.e. Test Dataset
            y_test_pred = model.predict(X_test)

            # Calculating Different Evaluation metrics for Training set predictions
            train_evaluation_artifact = get_regression_score(y_true=y_train,y_pred=y_train_pred)
            # Calculating Different Evaluation metrics for Test set predtions
            test_evaluation_artifact = get_regression_score(y_true=y_test,y_pred=y_test_pred)

            models_performance = {
                'model_train_mae':train_evaluation_artifact.mean_absolute_error,
                'model_train_rmse':train_evaluation_artifact.root_mean_squared_error,
                'model_train_r2_score':train_evaluation_artifact.r2_score,
                'model_test_mae':test_evaluation_artifact.mean_absolute_error,
                'model_test_rmse':test_evaluation_artifact.root_mean_squared_error,
                'model_test_r2_score':test_evaluation_artifact.r2_score,
                'best_params':gs.best_params_
            }
            report[list(models.keys())[i]] = models_performance
            plot_actual_vs_predicted(y_true=y_test, y_pred=y_test_pred,
                                     model_name=list(models.keys())[i], save_dir=plot_save_dir)
            plot_residuals(y_true=y_test, y_pred=y_test_pred,
                           model_name=list(models.keys())[i], save_dir=plot_save_dir)
            plot_feature_importance(model=model, feature_names=data_schema['final_columns'],
                                    model_name=list(models.keys())[i], save_dir=plot_save_dir)
            
        compare_models(reports=report, save_dir=plot_save_dir)
        write_yaml_file(file_path=model_trainer_config.model_evaluation_performance_report, content=report)

        return report
        
    except Exception as e:
        raise CustomException(e, sys)