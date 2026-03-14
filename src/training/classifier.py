"""
LightGBM classifier for forgery type classification
Implements Critical Fix #8: Configurable Confidence Threshold
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import json


class ForgeryClassifier:
    """
    LightGBM classifier for region-wise forgery classification
    
    Target classes:
    - 0: copy_move
    - 1: splicing
    - 2: text_substitution
    """
    
    CLASS_NAMES = ['copy_move', 'splicing', 'text_substitution']
    
    def __init__(self, config):
        """
        Initialize classifier
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # LightGBM parameters
        self.params = config.get('classifier.params', {
            'objective': 'multiclass',
            'num_class': 3,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        })
        
        # Critical Fix #8: Configurable confidence threshold
        self.confidence_threshold = config.get('classifier.confidence_threshold', 0.6)
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        
        # Feature importance
        self.feature_importance = None
        self.feature_names = None
    
    def train(self, 
              features: np.ndarray, 
              labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              validation_split: float = 0.2) -> Dict:
        """
        Train classifier
        
        Args:
            features: Feature matrix (N, D)
            labels: Class labels (N,)
            feature_names: Optional feature names
            validation_split: Validation split ratio
        
        Returns:
            Training metrics
        """
        print(f"Training LightGBM classifier")
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data (Critical Fix #7: Image-level splitting should be done upstream)
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels,
            test_size=validation_split,
            random_state=42,
            stratify=labels
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.params.get('n_estimators', 200),
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )
        
        # Store feature importance
        self.feature_names = feature_names
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        train_acc = (train_pred.argmax(axis=1) == y_train).mean()
        
        val_pred = self.model.predict(X_val)
        val_acc = (val_pred.argmax(axis=1) == y_val).mean()
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'num_features': features.shape[1],
            'num_samples': len(labels),
            'best_iteration': self.model.best_iteration
        }
        
        print(f"Training complete!")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Val accuracy: {val_acc:.4f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict forgery types
        
        Args:
            features: Feature matrix (N, D)
        
        Returns:
            predictions: Predicted class indices (N,)
            confidences: Prediction confidences (N,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probabilities = self.model.predict(features_scaled)
        
        # Get predictions and confidences
        predictions = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        
        return predictions, confidences
    
    def predict_with_filtering(self, 
                               features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence filtering
        
        Args:
            features: Feature matrix (N, D)
        
        Returns:
            predictions: Predicted class indices (N,)
            confidences: Prediction confidences (N,)
            valid_mask: Boolean mask for valid predictions (N,)
        """
        predictions, confidences = self.predict(features)
        
        # Critical Fix #8: Apply confidence threshold
        valid_mask = confidences >= self.confidence_threshold
        
        return predictions, confidences, valid_mask
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index"""
        return self.CLASS_NAMES[class_idx]
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top-k most important features
        
        Args:
            top_k: Number of features to return
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importance is None:
            return []
        
        # Sort by importance
        indices = np.argsort(self.feature_importance)[::-1][:top_k]
        
        result = []
        for idx in indices:
            name = self.feature_names[idx] if self.feature_names else f'feature_{idx}'
            importance = self.feature_importance[idx]
            result.append((name, importance))
        
        return result
    
    def save(self, save_dir: str):
        """
        Save model and scaler
        
        Args:
            save_dir: Directory to save model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        model_path = save_path / 'lightgbm_model.txt'
        self.model.save_model(str(model_path))
        
        # Save scaler
        scaler_path = save_path / 'scaler.joblib'
        joblib.dump(self.scaler, str(scaler_path))
        
        # Save metadata
        metadata = {
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.CLASS_NAMES,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        metadata_path = save_path / 'classifier_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Classifier saved to {save_path}")
    
    def load(self, load_dir: str):
        """
        Load model and scaler
        
        Args:
            load_dir: Directory to load from
        """
        load_path = Path(load_dir)
        
        # Load LightGBM model
        model_path = load_path / 'lightgbm_model.txt'
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load scaler
        scaler_path = load_path / 'scaler.joblib'
        self.scaler = joblib.load(str(scaler_path))
        
        # Load metadata
        metadata_path = load_path / 'classifier_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.confidence_threshold = metadata.get('confidence_threshold', 0.6)
        self.feature_names = metadata.get('feature_names')
        self.feature_importance = np.array(metadata.get('feature_importance', []))
        
        print(f"Classifier loaded from {load_path}")


def get_classifier(config) -> ForgeryClassifier:
    """Factory function for classifier"""
    return ForgeryClassifier(config)
