#!/usr/bin/env python3
"""
Binary Classifier Training
==========================

Train and evaluate the SeaClutterClassifier for binary classification
of clutter tracks vs actual target tracks.
"""

import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_tracker.models.deep_sort import SeaClutterClassifier


class BinaryClassificationTrainer:
    """Trainer for binary clutter/target classification."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            model: The classifier model
            device: Device to use ('auto', 'cpu', 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
        
        # Use weighted loss to handle class imbalance
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        print(f"Training on device: {self.device}")
    
    def compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights for handling imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        return torch.FloatTensor(class_weights).to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).long()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).long()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        return avg_loss, accuracy, f1
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              patience: int = 15,
              save_best: bool = True,
              model_save_path: str = None) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_best: Whether to save the best model
            model_save_path: Path to save the model
            
        Returns:
            training_results: Dictionary with training results
        """
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")
            
            # Early stopping and model saving
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                
                if save_best and model_save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_f1': val_f1,
                        'history': self.history
                    }, model_save_path)
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val F1: {best_val_f1:.3f})")
                break
        
        training_results = {
            'best_val_f1': best_val_f1,
            'total_epochs': epoch + 1,
            'history': self.history
        }
        
        print(f"Training completed! Best validation F1: {best_val_f1:.3f}")
        
        return training_results


def load_binary_classification_data(data_dir: str = 'data/binary_classification') -> Tuple:
    """Load preprocessed binary classification data."""
    data_path = Path(data_dir)
    
    # Load training data
    with open(data_path / 'train_data_scaled.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    # Load validation data
    with open(data_path / 'val_data_scaled.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    # Load test data
    with open(data_path / 'test_data_scaled.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Load scaler and metadata
    with open(data_path / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(data_path / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    return train_data, val_data, test_data, scaler, dataset_info


def create_data_loaders(train_data: Dict, val_data: Dict, test_data: Dict,
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch data loaders."""
    
    # Convert to tensors
    X_train = torch.FloatTensor(train_data['X'])
    y_train = torch.LongTensor(train_data['y'])
    
    X_val = torch.FloatTensor(val_data['X'])
    y_val = torch.LongTensor(val_data['y'])
    
    X_test = torch.FloatTensor(test_data['X'])
    y_test = torch.LongTensor(test_data['y'])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: torch.device) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).long()
            
            outputs = model(batch_x)
            probabilities = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy()[:, 1])  # Probability of class 1 (target)
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    
    try:
        auc = roc_auc_score(all_targets, all_probabilities)
    except ValueError:
        auc = 0.0  # In case of single class in test set
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=['Clutter', 'Target'],
                                       output_dict=True)
    
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }
    
    return evaluation_results


def create_evaluation_plots(evaluation_results: Dict[str, Any], 
                          training_history: Dict[str, List],
                          output_dir: Path):
    """Create comprehensive evaluation plots."""
    
    # Training history plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(training_history['train_loss'], label='Training Loss')
    axes[0, 0].plot(training_history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(training_history['train_acc'], label='Training Accuracy')
    axes[0, 1].plot(training_history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[1, 0].plot(training_history['train_f1'], label='Training F1')
    axes[1, 0].plot(training_history['val_f1'], label='Validation F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = evaluation_results['confusion_matrix']
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
    
    classes = ['Clutter', 'Target']
    tick_marks = np.arange(len(classes))
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_xticklabels(classes)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC and Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    if 'targets' in evaluation_results and 'probabilities' in evaluation_results:
        fpr, tpr, _ = roc_curve(evaluation_results['targets'], evaluation_results['probabilities'])
        axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {evaluation_results["auc"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            evaluation_results['targets'], evaluation_results['probabilities']
        )
        axes[1].plot(recall_curve, precision_curve, 
                    label=f'PR Curve (F1 = {evaluation_results["f1_score"]:.3f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Evaluation plots saved")


def main():
    """Main training function."""
    print("Binary Classifier Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 15,
        'input_dim': 8,
        'hidden_dims': [64, 32, 16]
    }
    
    # Output directory
    output_dir = Path('models/binary_classifier')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        print("Loading binary classification data...")
        train_data, val_data, test_data, scaler, dataset_info = load_binary_classification_data()
        
        print(f"Data loaded:")
        print(f"  Train: {len(train_data['X'])} samples")
        print(f"  Val:   {len(val_data['X'])} samples") 
        print(f"  Test:  {len(test_data['X'])} samples")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, config['batch_size']
        )
        
        # Create model
        print("Creating model...")
        model = SeaClutterClassifier(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims']
        )
        
        # Create trainer
        trainer = BinaryClassificationTrainer(
            model=model,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Train model
        print("Starting training...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            patience=config['patience'],
            save_best=True,
            model_save_path=str(output_dir / 'best_model.pth')
        )
        
        # Load best model for evaluation
        checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        print("Evaluating on test set...")
        evaluation_results = evaluate_model(model, test_loader, trainer.device)
        
        # Print results
        print("\n" + "=" * 50)
        print("Training Results")
        print("=" * 50)
        print(f"Best Validation F1: {training_results['best_val_f1']:.3f}")
        print(f"Total Epochs: {training_results['total_epochs']}")
        
        print("\nTest Set Evaluation:")
        print(f"  Accuracy:  {evaluation_results['accuracy']:.3f}")
        print(f"  Precision: {evaluation_results['precision']:.3f}")
        print(f"  Recall:    {evaluation_results['recall']:.3f}")
        print(f"  F1 Score:  {evaluation_results['f1_score']:.3f}")
        print(f"  AUC:       {evaluation_results['auc']:.3f}")
        
        print("\nConfusion Matrix:")
        print("           Predicted")
        print("         Clutter Target")
        cm = evaluation_results['confusion_matrix']
        print(f"Clutter     {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"Target      {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Save results
        results_summary = {
            'config': config,
            'training_results': training_results,
            'evaluation_results': {
                'accuracy': evaluation_results['accuracy'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'f1_score': evaluation_results['f1_score'],
                'auc': evaluation_results['auc'],
                'confusion_matrix': evaluation_results['confusion_matrix'].tolist(),
                'classification_report': evaluation_results['classification_report']
            },
            'dataset_info': dataset_info
        }
        
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Create evaluation plots
        create_evaluation_plots(
            evaluation_results, 
            training_results['history'], 
            output_dir
        )
        
        print(f"\nModel and results saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Review training plots and metrics")
        print("  2. Use the trained model for real-time classification")
        print("  3. Integrate with the tracking system")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()