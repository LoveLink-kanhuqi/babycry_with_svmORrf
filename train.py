import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV  # 添加这个导入
import joblib
import os
import random
import torch
import json
from feature_extractor import FeatureExtractor, DataProcessor
import argparse
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CryDetectionModel:
    def __init__(self, model_type='svm', **kwargs):
        self.model_type = model_type
        if model_type == 'svm':
            # 确保SVM启用概率估计
            kwargs['probability'] = True
            self.model = SVC(**kwargs)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.feature_extractor = FeatureExtractor()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model and return training history."""
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Calculate training and validation scores
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        return {
            'train_score': train_score,
            'val_score': val_score
        }
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates for samples."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # 如果模型不支持概率估计，返回决策函数值
            if hasattr(self.model, 'decision_function'):
                decisions = self.model.decision_function(X)
                # 将决策函数值转换为伪概率 (使用sigmoid函数)
                return 1 / (1 + np.exp(-decisions))
            else:
                # 如果既没有概率也没有决策函数，返回二进制预测
                return self.model.predict(X).astype(float)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(X_test)
        y_prob = self.get_probabilities(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        }
        
    def predict(self, audio_path: str) -> Tuple[int, float]:
        """Predict for a single audio file."""
        features = self.feature_extractor.extract_features(audio_path)
        features = features.reshape(1, -1)
        features = self.feature_extractor.transform_features(features)
        
        prediction = self.model.predict(features)[0]
        probability = self.get_probabilities(features)[0]
            
        return prediction, probability
        
    def save(self, model_path: str):
        """Save the model and feature extractor scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler separately
        model_file = model_path
        scaler_file = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
        
        joblib.dump(self.model, model_file)
        self.feature_extractor.save_scaler(scaler_file)
        
    @classmethod
    def load(cls, model_path: str):
        """Load a saved model and scaler."""
        # Load model
        model_file = model_path
        scaler_file = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
        
        loaded_model = joblib.load(model_file)
        
        # Create instance and set attributes
        instance = cls(model_type='svm' if isinstance(loaded_model, SVC) else 'rf')
        instance.model = loaded_model
        
        # Load scaler
        instance.feature_extractor.load_scaler(scaler_file)
        
        return instance

def plot_training_metrics(history: Dict, evaluation: Dict, model_path: str):
    """Plot and save training metrics and evaluation results."""
    # Create directory for plots
    plot_dir = os.path.join(os.path.dirname(model_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Training History
    ax1 = plt.subplot(231)
    ax1.plot(history['train_score'], 'b-', label='Training Accuracy')
    ax1.plot(history['val_score'], 'r-', label='Validation Accuracy')
    ax1.set_title('Training History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(232)
    conf_matrix = evaluation['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # 3. Classification Report
    ax3 = plt.subplot(233)
    report = evaluation['classification_report']
    report_df = pd.DataFrame(report).transpose()
    # Filter out unnecessary rows and columns
    report_df = report_df.iloc[:-3]  # Remove avg rows
    cell_text = []
    for idx in report_df.index:
        row = report_df.loc[idx]
        cell_text.append([f'{row["precision"]:.3f}', 
                         f'{row["recall"]:.3f}', 
                         f'{row["f1-score"]:.3f}'])
    
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text,
                     rowLabels=report_df.index,
                     colLabels=['Precision', 'Recall', 'F1-Score'],
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax3.set_title('Classification Report')
    
    # 4. ROC Curve
    if 'roc_curve' in evaluation:
        ax4 = plt.subplot(234)
        fpr = evaluation['roc_curve']['fpr']
        tpr = evaluation['roc_curve']['tpr']
        roc_auc = evaluation['roc_curve']['auc']
        ax4.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax4.set_xlim([0.0, 1.0])
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend(loc="lower right")
        ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(plot_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Train cry detection model')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'rf'],
                        help='Type of model to train')
    parser.add_argument('--output_path', type=str, default='models/cry_detection_model.joblib',
                        help='Path to save the trained model')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.random_seed)
    print(f"Using random seed: {args.random_seed}")

    # Split data
    data_processor = DataProcessor()
    train_df, val_df, test_df = data_processor.split_data(args.data_csv, random_state=args.random_seed)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract features for all sets
    print("Extracting features...")
    X_train = feature_extractor.extract_features_batch(train_df['filename'].values)
    X_val = feature_extractor.extract_features_batch(val_df['filename'].values)
    X_test = feature_extractor.extract_features_batch(test_df['filename'].values)
    
    # Fit and transform features
    feature_extractor.fit_scaler(X_train)
    X_train = feature_extractor.transform_features(X_train)
    X_val = feature_extractor.transform_features(X_val)
    X_test = feature_extractor.transform_features(X_test)
    
    # Prepare labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Define hyperparameter grid
    param_grid = {
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    # Perform grid search
    print("Performing grid search...")
    if args.model_type == 'svm':
        base_model = SVC(probability=True, random_state=args.random_seed)
    else:
        base_model = RandomForestClassifier(random_state=args.random_seed)
        
    grid_search = GridSearchCV(
        base_model,
        param_grid[args.model_type],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Train final model with best parameters
    print("Training final model...")
    model = CryDetectionModel(model_type=args.model_type, **grid_search.best_params_)
    model.feature_extractor = feature_extractor
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation = model.evaluate(X_test, y_test)
    
    # Print metrics
    print("\nClassification Report:")
    print(pd.DataFrame(evaluation['classification_report']).transpose())
    
    print("\nConfusion Matrix:")
    print(evaluation['confusion_matrix'])
    
    # Plot and save all metrics
    plot_training_metrics(history, evaluation, args.output_path)
    
    # Save model
    model.save(args.output_path)
    print(f"\nModel saved to {args.output_path}")

    # Save experiment configuration
    config = {
        'random_seed': args.random_seed,
        'model_type': args.model_type,
        'best_parameters': grid_search.best_params_,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    config_path = os.path.join(os.path.dirname(args.output_path), 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    main()