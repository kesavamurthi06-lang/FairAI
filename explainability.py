"""
Explainability Module for FairAI Guardian
========================================
Production-quality Explainable AI (XAI) engine using SHAP for model interpretability.

Author: FairAI Guardian Team
Version: 1.0.0
"""

import logging
import os
import warnings
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.base import is_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fairai_explainability.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Comprehensive Explainable AI engine for FairAI Guardian using SHAP.
    
    Provides global and local model interpretability for binary classification models
    with publication-quality visualizations and explanation artifacts.
    """
    
    def __init__(self):
        self.model: Optional[Any] = None
        """Trained model"""
        self.explainer: Optional[Any] = None
        """SHAP explainer"""
        self.shap_values: Optional[np.ndarray] = None
        """Computed SHAP values"""
        self.feature_names: Optional[list] = None
        """Feature names"""
        self.background_data: Optional[np.ndarray] = None
        """Background dataset for SHAP"""
        
        logger.info("ModelExplainer initialized")
    
    def fit_explainer(
        self,
        model: Any,
        X_train: pd.DataFrame,
        model_type: str = "auto",
        n_background: int = 100
    ) -> None:
        """
        Initialize SHAP explainer for the given model.
        
        Args:
            model: Trained model
            X_train: Training data for background dataset
            model_type: 'auto', 'tree', 'linear', 'kernel'
            n_background: Number of background samples (for kernel explainer)
        """
        if X_train.empty:
            raise ValueError("X_train cannot be empty")
        
        self.model = model
        self.feature_names = X_train.columns.tolist()
        
        # Create background dataset
        if n_background >= len(X_train):
            self.background_data = X_train.values
        else:
            self.background_data = shap.kmeans(X_train.values, n_background).data
        
        # Auto-detect model type or use specified type
        if model_type == "auto":
            if hasattr(model, 'tree_') or 'RandomForest' in str(type(model)):
                model_type = "tree"
            elif isinstance(model, (shap.LinearExplainer,)):
                model_type = "linear"
            else:
                model_type = "kernel"
        
        try:
            if model_type == "tree":
                self.explainer = shap.TreeExplainer(model)
                logger.info("TreeExplainer initialized")
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(model, self.background_data)
                logger.info("LinearExplainer initialized")
            elif model_type == "kernel":
                self.explainer = shap.KernelExplainer(
                    model.predict_proba if is_classifier(model) else model.predict,
                    self.background_data
                )
                logger.info("KernelExplainer initialized")
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_type} explainer: {str(e)}")
            raise ValueError(f"Could not initialize explainer for model type '{model_type}': {str(e)}")
    
    def compute_shap_values(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute SHAP values for the input dataset.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: SHAP values (n_samples, n_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")
        
        if list(X.columns) != self.feature_names:
            raise ValueError("Input features must match training feature names")
        
        try:
            # For binary classification, get SHAP values for positive class
            shap_values = self.explainer.shap_values(X.values)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # TreeExplainer returns list for multiclass
                self.shap_values = shap_values[1]  # Positive class
            else:
                self.shap_values = shap_values
            
            logger.info(f"SHAP values computed for {len(X)} samples")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            raise RuntimeError(f"Failed to compute SHAP values: {str(e)}")
    
    def get_global_feature_importance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute global feature importance using mean absolute SHAP values.
        
        Args:
            X: Dataset to compute SHAP values
            
        Returns:
            pd.DataFrame: Feature importance ranked by mean |SHAP|
        """
        self.compute_shap_values(X)
        
        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_absolute_shap_value': importance
        }).sort_values('mean_absolute_shap_value', ascending=False)
        
        logger.info("Global feature importance computed")
        return importance_df
    
    def explain_prediction(
        self,
        X_instance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate local explanation for a single prediction.
        
        Args:
            X_instance: Single instance (1 row DataFrame)
            
        Returns:
            pd.DataFrame: Local feature contributions sorted by impact
        """
        if len(X_instance) != 1:
            raise ValueError("X_instance must contain exactly 1 row")
        
        shap_vals = self.compute_shap_values(X_instance)
        instance_shap = shap_vals[0]
        
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': X_instance.iloc[0].values,
            'shap_value': instance_shap,
            'impact_direction': ['Positive' if v > 0 else 'Negative' for v in instance_shap]
        })
        
        explanation['abs_shap_value'] = np.abs(explanation['shap_value'])
        explanation = explanation.sort_values('abs_shap_value', ascending=False).drop('abs_shap_value', axis=1)
        
        logger.info("Local explanation generated")
        return explanation
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Generate SHAP summary plot (beeswarm).
        
        Args:
            X: Dataset
            save_path: Path to save plot
            show: Display plot
        """
        self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X.values,
            feature_names=self.feature_names,
            show=False
        )
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        max_features: int = 10,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Generate global feature importance bar plot.
        
        Args:
            X: Dataset
            max_features: Maximum number of features to show
            save_path: Path to save plot
            show: Display plot
        """
        importance_df = self.get_global_feature_importance(X)
        top_features = importance_df.head(max_features)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_features['mean_absolute_shap_value'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Global Feature Importance (Top 10)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', va='center')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_waterfall(
        self,
        X_instance: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Generate waterfall plot for single instance explanation.
        
        Args:
            X_instance: Single instance (1 row)
            save_path: Path to save plot
            show: Display plot
        """
        if len(X_instance) != 1:
            raise ValueError("X_instance must contain exactly 1 row")
        
        self.compute_shap_values(X_instance)
        shap_values_instance = self.shap_values[0]
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_instance,
                base_values=self.explainer.expected_value,
                data=X_instance.iloc[0].values,
                feature_names=self.feature_names
            ),
            show=False
        )
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_force(
        self,
        X_instance: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Generate SHAP force plot for single instance.
        
        Args:
            X_instance: Single instance (1 row)
            save_path: Path to save plot
            show: Display plot
        """
        if len(X_instance) != 1:
            raise ValueError("X_instance must contain exactly 1 row")
        
        self.compute_shap_values(X_instance)
        shap_values_instance = self.shap_values[0]
        
        shap.force_plot(
            self.explainer.expected_value,
            shap_values_instance,
            X_instance.iloc[0].values,
            feature_names=self.feature_names,
            show=False,
            matplotlib=True
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def save_explainer(
        self,
        file_path: str
    ) -> None:
        """
        Save explainer and model to disk.
        
        Args:
            file_path: Path to save explainer
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")
        
        explainer_data = {
            'model': self.model,
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'background_data': self.background_data
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(explainer_data, file_path)
        logger.info(f"Explainer saved to {file_path}")
    
    def load_explainer(
        self,
        file_path: str
    ) -> None:
        """
        Load explainer from disk.
        
        Args:
            file_path: Path to saved explainer
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Explainer file not found: {file_path}")
        
        explainer_data = joblib.load(file_path)
        self.model = explainer_data['model']
        self.explainer = explainer_data['explainer']
        self.feature_names = explainer_data['feature_names']
        self.background_data = explainer_data['background_data']
        
        logger.info(f"Explainer loaded from {file_path}")
    
    def get_model_prediction(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and probabilities.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call fit_explainer() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        return predictions, probabilities


def main():
    """Example usage of ModelExplainer."""
    try:
        # Generate sample data and train a model
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_df, y_train)
        
        # Initialize explainer
        explainer = ModelExplainer()
        explainer.fit_explainer(model, X_train_df)
        
        # Global explanations
        importance_df = explainer.get_global_feature_importance(X_test_df)
        print("Top 5 Most Important Features:\n", importance_df.head())
        
        # Local explanation
        instance_explanation = explainer.explain_prediction(X_test_df.iloc[[0]])
        print("\nLocal Explanation (Instance 0):\n", instance_explanation.head())
        
        # Generate plots
        os.makedirs("reports", exist_ok=True)
        explainer.plot_summary(X_test_df, save_path="reports/shap_summary.png")
        explainer.plot_feature_importance(X_test_df, save_path="reports/feature_importance.png")
        explainer.plot_waterfall(X_test_df.iloc[[0]], save_path="reports/waterfall_0.png")
        explainer.plot_force(X_test_df.iloc[[0]], save_path="reports/force_0.png")
        
        # Save explainer
        explainer.save_explainer("models/explainer.pkl")
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()