"""
Data Preprocessing Module for FairAI Guardian
============================================
A production-quality data ingestion and preprocessing pipeline designed for AI fairness analysis.

Author: FairAI Guardian Team
Version: 1.0.0
"""

import logging
import os
import pickle
import warnings
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fairai_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for FairAI Guardian project.
    
    Handles data loading, inspection, missing value imputation, feature encoding,
    scaling, and stratified train-test splitting with sensitive attribute preservation.
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.numerical_features: list = []
        self.categorical_features: list = []
        self.binary_features: list = []
        self.target_col: Optional[str] = None
        self.sensitive_col: Optional[str] = None
        
        # Preprocessing pipelines
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.preprocessor: Optional[ColumnTransformer] = None
        
        # Data splits
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.sensitive_train: Optional[np.ndarray] = None
        self.sensitive_test: Optional[np.ndarray] = None
        
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file with robust error handling.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file does not exist
            pd.errors.EmptyDataError: If file is empty
            ValueError: If file format is unsupported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Unsupported file format or read error: {str(e)}")
    
    def inspect_data(self) -> Dict[str, Any]:
        """
        Display comprehensive dataset information.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'summary_stats': self.df.describe(include='all').to_dict()
        }
        
        logger.info(f"Dataset inspection completed:")
        logger.info(f"  - Shape: {info['shape']}")
        logger.info(f"  - Columns: {len(info['columns'])}")
        logger.info(f"  - Missing values: {sum(info['missing_values'].values())}")
        
        return info
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values using domain-appropriate imputation strategies.
        - Numerical: median
        - Categorical: mode
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Identify feature types for imputation
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute numerical columns with median
        for col in numerical_cols:
            median_val = self.df[col].median()
            self.df[col].fillna(median_val, inplace=True)
            logger.debug(f"Filled {col} NaNs with median: {median_val}")
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            mode_val = self.df[col].mode()
            if not mode_val.empty:
                self.df[col].fillna(mode_val.iloc[0], inplace=True)
                logger.debug(f"Filled {col} NaNs with mode: {mode_val.iloc[0]}")
            else:
                self.df[col].fillna('Unknown', inplace=True)
        
        logger.info("Missing value imputation completed")
    
    def identify_feature_types(self) -> Tuple[list, list, list]:
        """
        Automatically identify numerical, categorical, and binary features.
        
        Returns:
            Tuple[list, list, list]: (numerical_features, categorical_features, binary_features)
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Remove target and sensitive columns if specified
        cols_to_exclude = []
        if self.target_col and self.target_col in self.df.columns:
            cols_to_exclude.append(self.target_col)
        if self.sensitive_col and self.sensitive_col in self.df.columns:
            cols_to_exclude.append(self.sensitive_col)
        
        feature_cols = [col for col in self.df.columns if col not in cols_to_exclude]
        
        self.numerical_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identify binary categorical features (2 unique values)
        self.binary_features = []
        self.categorical_features = []
        for col in object_cols:
            unique_count = self.df[col].nunique()
            if unique_count == 2:
                self.binary_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"Feature types identified:")
        logger.info(f"  - Numerical: {len(self.numerical_features)}")
        logger.info(f"  - Categorical: {len(self.categorical_features)}")
        logger.info(f"  - Binary: {len(self.binary_features)}")
        
        return self.numerical_features, self.categorical_features, self.binary_features
    
    def encode_features(self) -> None:
        """
        Encode categorical features:
        - Binary: Label Encoding
        - Multi-class: One-Hot Encoding
        """
        if not self.categorical_features and not self.binary_features:
            logger.warning("No categorical features to encode")
            return
        
        # Label encode binary features
        for col in self.binary_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            logger.debug(f"Label encoded binary feature: {col}")
        
        # One-hot encode multi-class categorical features
        if self.categorical_features:
            ohe_df = pd.get_dummies(self.df[self.categorical_features], prefix=self.categorical_features)
            self.df = pd.concat([self.df.drop(self.categorical_features, axis=1), ohe_df], axis=1)
            logger.info(f"One-hot encoded {len(self.categorical_features)} categorical features")
    
    def scale_features(self) -> None:
        """
        Scale numerical features using StandardScaler.
        """
        if not self.numerical_features:
            logger.warning("No numerical features to scale")
            return
        
        self.df[self.numerical_features] = self.scaler.fit_transform(self.df[self.numerical_features])
        logger.info(f"Scaled {len(self.numerical_features)} numerical features")
    
    def prepare_data(self, target_col: str, sensitive_col: str) -> Tuple[np.ndarray, np.ndarray, 
                                                                        np.ndarray, np.ndarray, 
                                                                        np.ndarray, np.ndarray]:
        """
        Prepare data for training with stratified split and sensitive attribute preservation.
        
        Args:
            target_col (str): Target variable column name
            sensitive_col (str): Sensitive attribute column name (e.g., 'gender', 'race')
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (X_train, X_test, y_train, y_test, sensitive_train, sensitive_test)
            
        Raises:
            ValueError: If columns don't exist or data not ready
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded or preprocessed. Run preprocessing steps first.")
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        if sensitive_col not in self.df.columns:
            raise ValueError(f"Sensitive column '{sensitive_col}' not found")
        
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        
        # Prepare features (exclude target and sensitive)
        feature_cols = [col for col in self.df.columns 
                       if col not in [target_col, sensitive_col]]
        X = self.df[feature_cols]
        y = self.df[target_col]
        sensitive = self.df[sensitive_col]
        
        # Stratified train-test split
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.sensitive_train, self.sensitive_test = train_test_split(
            X, y, sensitive,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        logger.info("Data preparation completed:")
        logger.info(f"  - Training set: {self.X_train.shape}")
        logger.info(f"  - Test set: {self.X_test.shape}")
        
        return (self.X_train, self.X_test, self.y_train, self.y_test, 
                self.sensitive_train, self.sensitive_test)
    
    def save_pipeline(self, file_path: str) -> None:
        """
        Save the fitted preprocessing pipeline.
        
        Args:
            file_path (str): Path to save the pipeline
        """
        pipeline_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'binary_features': self.binary_features
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {file_path}")
    
    def load_pipeline(self, file_path: str) -> None:
        """
        Load a saved preprocessing pipeline.
        
        Args:
            file_path (str): Path to the saved pipeline
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.scaler = pipeline_data['scaler']
        self.label_encoders = pipeline_data['label_encoders']
        self.numerical_features = pipeline_data['numerical_features']
        self.categorical_features = pipeline_data['categorical_features']
        self.binary_features = pipeline_data['binary_features']
        
        logger.info(f"Pipeline loaded from {file_path}")


def main():
    """Example usage of DataPreprocessor."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load data (replace with your CSV path)
        # preprocessor.load_data('adult.csv')
        
        # Inspect data
        # info = preprocessor.inspect_data()
        
        # Handle missing values and preprocess
        # preprocessor.handle_missing_values()
        # preprocessor.identify_feature_types()
        # preprocessor.encode_features()
        # preprocessor.scale_features()
        
        # Prepare data for fairness analysis
        # X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = \
        #     preprocessor.prepare_data('income', 'gender')
        
        # Save pipeline
        # preprocessor.save_pipeline('fairai_pipeline.pkl')
        
        logger.info("Example execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()