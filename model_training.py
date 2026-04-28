
"""
Model Training Module for FairAI Guardian
========================================
Production-quality machine learning training engine for binary classification
with model comparison, evaluation, and persistence.

Author: FairAI Guardian Team
Version: 1.0.0
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fairai_model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training and evaluation engine for FairAI Guardian.
    
    Supports Logistic Regression, Random Forest, and Gradient Boosting with
    automated comparison, evaluation, and model persistence.
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        """Dictionary of trained models {name: model}"""
        self.results: Dict[str, Dict[str, float]] = {}
        """Dictionary of evaluation results {model_name: metrics}"""
        self.feature_names: Optional[List[str]] = None
        """Stored feature names for validation"""
        
        logger.info("ModelTrainer initialized")
    
    def _validate_inputs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> None:
        """
        Validate training and testing data.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        
        if X_train.shape[0] != len(y_train):
            raise ValueError("X_train and y_train must have matching number of samples")
        
        if X_test is not None and y_test is not None:
            if X_test.shape[0] != len(y_test):
                raise ValueError("X_test and y_test must have matching number of samples")
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("X_train and X_test must have same number of features")
        
        if not np.isin(y_train.unique(), [0, 1]).all():
            raise ValueError("y_train must contain only binary labels (0/1)")
        
        self.feature_names = X_train.columns.tolist()
        logger.debug(f"Input validation passed: {X_train.shape[0]} train samples")
    
    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics.
        """
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
            
            # Store additional diagnostics
            self.results[model_name]['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            self.results[model_name]['classification_report'] = classification_report(
                y_test, y_pred, output_dict=True
            )
            
            logger.info(f"{model_name} evaluation: F1={metrics['f1_score']:.4f}, AUC={metrics.get('roc_auc', 0):.4f}")
            return metrics
            
        except NotFittedError:
            raise ValueError(f"Model {model_name} is not fitted. Train model first.")
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional hyperparameters
            
        Returns:
            LogisticRegression: Trained model
        """
        self._validate_inputs(X_train, y_train)
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **kwargs
        )
        
        logger.info("Training Logistic Regression...")
        model.fit(X_train, y_train)
        
        model_name = 'logistic_regression'
        self.models[model_name] = model
        self.results[model_name] = {'model_type': 'LogisticRegression'}
        
        logger.info(f"Logistic Regression trained successfully")
        return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest Classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional hyperparameters
            
        Returns:
            RandomForestClassifier: Trained model
        """
        self._validate_inputs(X_train, y_train)
        
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            **kwargs
        )
        
        logger.info("Training Random Forest...")
        model.fit(X_train, y_train)
        
        model_name = 'random_forest'
        self.models[model_name] = model
        self.results[model_name] = {'model_type': 'RandomForest'}
        
        logger.info(f"Random Forest trained successfully")
        return model
    
    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting Classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional hyperparameters
            
        Returns:
            GradientBoostingClassifier: Trained model
        """
        self._validate_inputs(X_train, y_train)
        
        model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            **kwargs
        )
        
        logger.info("Training Gradient Boosting...")
        model.fit(X_train, y_train)
        
        model_name = 'gradient_boosting'
        self.models[model_name] = model
        self.results[model_name] = {'model_type': 'GradientBoosting'}
        
        logger.info(f"Gradient Boosting trained successfully")
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name to store results under
            
        Returns:
            Dict[str, float]: Model evaluation metrics
        """
        self._validate_inputs(pd.DataFrame(), pd.Series(), X_test, y_test)
        
        if model_name is None:
            model_name = f"model_{id(model)}"
        
        metrics = self._evaluate_model(model, X_test, y_test, model_name)
        self.results[model_name].update(metrics)
        
        return metrics
    
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Train and compare all supported models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Model comparison table sorted by F1-score
        """
        self._validate_inputs(X_train, y_train, X_test, y_test)
        
        # Clear previous results
        self.models.clear()
        self.results.clear()
        
        # Train all models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate all models
        comparison_data = []
        for model_name, model in self.models.items():
            metrics = self._evaluate_model(model, X_test, y_test, model_name)
            comparison_data.append({
                'model': model_name.replace('_', ' ').title(),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics.get('roc_auc', 0.0)
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('f1_score', ascending=False)
        logger.info(f"Model comparison completed:\n{comparison_df}")
        
        return comparison_df
    
    def get_best_model(self, metric: str = "f1_score") -> Tuple[str, Any]:
        """
        Select best model based on specified metric.
        
        Args:
            metric: Metric to optimize ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
            
        Returns:
            Tuple[str, Any]: (best_model_name, best_model)
        """
        if not self.results:
            raise ValueError("No models trained. Run compare_models() or train models first.")
        
        if metric not in self.results[list(self.results.keys())[0]]:
            available_metrics = list(next(iter(self.results.values())).keys())
            raise ValueError(f"Metric '{metric}' not available. Choose from: {available_metrics}")
        
        best_model_name = max(
            self.results.keys(),
            key=lambda x: self.results[x].get(metric, 0)
        )
        
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name][metric]
        
        logger.info(f"Best model: {best_model_name} ({metric} = {best_score:.4f})")
        return best_model_name, best_model
    
    def predict(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            np.ndarray: Binary predictions
        """
        if self.feature_names is not None and list(X.columns) != self.feature_names:
            logger.warning("Feature mismatch detected")
        
        return model.predict(X)
    
    def predict_proba(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            np.ndarray: Probability predictions [P(0), P(1)]
        """
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model does not support predict_proba")
        
        if self.feature_names is not None and list(X.columns) != self.feature_names:
            logger.warning("Feature mismatch detected")
        
        return model.predict_proba(X)
    
    def save_model(
        self,
        model: Any,
        file_path: str
    ) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            file_path: Path to save model
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(
        self,
        file_path: str
    ) -> Any:
        """
        Load trained model from disk.
        
        Args:
            file_path: Path to saved model
            
        Returns:
            Any: Loaded model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
    
    def get_model_results(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete results for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Complete results including metrics and diagnostics
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results")
        
        return self.results[model_name]


def main():
    """Example usage of ModelTrainer."""
    try:
        # Generate sample data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                                 n_redundant=5, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Compare all models
        comparison = trainer.compare_models(X_train_df, y_train_series, X_test_df, y_test_series)
        print("Model Comparison:\n", comparison.round(4))
        
        # Get best model
        best_name, best_model = trainer.get_best_model('f1_score')
        print(f"\nBest model: {best_name}")
        
        # Save best model
        trainer.save_model(best_model, "models/best_model.pkl")
        
        # Load and test
        loaded_model = trainer.load_model("models/best_model.pkl")
        metrics = trainer.evaluate_model(loaded_model, X_test_df, y_test_series, "loaded_model")
        print("Loaded model metrics:", metrics)
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()
