#!/usr/bin/env python3
"""
Binary Classification Inference
===============================

Real-time inference for clutter vs target classification using the trained
SeaClutterClassifier model. This script can be integrated with the tracking
system for real-time filtering of clutter tracks.
"""

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_tracker.models.deep_sort import SeaClutterClassifier
from prepare_binary_classification_data import extract_detection_features


class BinaryClutterClassifier:
    """Real-time binary clutter/target classifier."""
    
    def __init__(self, 
                 model_path: str = 'models/binary_classifier/best_model.pth',
                 scaler_path: str = 'data/binary_classification/scaler.pkl',
                 config_path: str = 'models/binary_classifier/training_results.json',
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        """
        Initialize the binary classifier.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
            config_path: Path to training configuration
            device: Device to use ('auto', 'cpu', 'cuda')
            confidence_threshold: Threshold for binary classification
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.confidence_threshold = confidence_threshold
        
        # Load scaler
        self.scaler = self._load_scaler(scaler_path)
        
        # Load model configuration
        self.config = self._load_config(config_path)
        
        # Initialize and load model
        self.model = self._load_model(model_path)
        
        print(f"Binary classifier loaded on device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def _load_scaler(self, scaler_path: str):
        """Load the feature scaler."""
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        except FileNotFoundError:
            print(f"Warning: Scaler not found at {scaler_path}. Using no scaling.")
            return None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config['config']
        except FileNotFoundError:
            print(f"Warning: Config not found at {config_path}. Using default.")
            return {
                'input_dim': 8,
                'hidden_dims': [64, 32, 16]
            }
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model."""
        # Create model
        model = SeaClutterClassifier(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config['hidden_dims']
        )
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']} "
                  f"with validation F1: {checkpoint['val_f1']:.3f}")
        except FileNotFoundError:
            print(f"Warning: Model not found at {model_path}. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_detection(self, detection: Dict) -> np.ndarray:
        """
        Preprocess a single detection for classification.
        
        Args:
            detection: Detection dictionary with radar features
            
        Returns:
            features: Preprocessed feature vector
        """
        # Extract features
        features = extract_detection_features(detection)
        
        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        return features
    
    def classify_detection(self, detection: Dict) -> Tuple[int, float, Dict[str, Any]]:
        """
        Classify a single detection as clutter (0) or target (1).
        
        Args:
            detection: Detection dictionary
            
        Returns:
            prediction: 0 for clutter, 1 for target
            confidence: Confidence score [0, 1]
            details: Additional classification details
        """
        # Preprocess
        features = self.preprocess_detection(detection)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            target_prob = probabilities[0, 1].item()  # Probability of being a target
            prediction = 1 if target_prob > self.confidence_threshold else 0
            confidence = target_prob if prediction == 1 else (1 - target_prob)
        
        details = {
            'target_probability': target_prob,
            'clutter_probability': probabilities[0, 0].item(),
            'raw_logits': outputs[0].cpu().numpy().tolist(),
            'features_used': features.tolist(),
            'threshold_used': self.confidence_threshold
        }
        
        return prediction, confidence, details
    
    def classify_detections_batch(self, detections: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """
        Classify multiple detections in batch for efficiency.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            results: List of (prediction, confidence, details) tuples
        """
        if not detections:
            return []
        
        # Preprocess all detections
        features_batch = []
        for detection in detections:
            features = self.preprocess_detection(detection)
            features_batch.append(features)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(np.array(features_batch)).to(self.device)
        
        # Batch inference
        results = []
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            for i, detection in enumerate(detections):
                target_prob = probabilities[i, 1].item()
                prediction = 1 if target_prob > self.confidence_threshold else 0
                confidence = target_prob if prediction == 1 else (1 - target_prob)
                
                details = {
                    'target_probability': target_prob,
                    'clutter_probability': probabilities[i, 0].item(),
                    'raw_logits': outputs[i].cpu().numpy().tolist(),
                    'features_used': features_batch[i].tolist(),
                    'threshold_used': self.confidence_threshold
                }
                
                results.append((prediction, confidence, details))
        
        return results
    
    def filter_targets_from_frame(self, frame_data: Dict, 
                                 min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Filter target tracks from a frame, removing predicted clutter.
        
        Args:
            frame_data: Frame data dictionary with 'all_detections'
            min_confidence: Minimum confidence for keeping a target
            
        Returns:
            filtered_data: Frame data with filtered detections
        """
        all_detections = frame_data.get('all_detections', [])
        
        if not all_detections:
            return frame_data.copy()
        
        # Classify all detections
        classification_results = self.classify_detections_batch(all_detections)
        
        # Filter based on predictions and confidence
        filtered_detections = []
        classification_metadata = []
        
        for detection, (prediction, confidence, details) in zip(all_detections, classification_results):
            # Keep if predicted as target with sufficient confidence
            if prediction == 1 and confidence >= min_confidence:
                # Add classification metadata to detection
                detection_copy = detection.copy()
                detection_copy['classification'] = {
                    'predicted_class': 'target',
                    'confidence': confidence,
                    'target_probability': details['target_probability']
                }
                filtered_detections.append(detection_copy)
            
            # Store classification metadata
            classification_metadata.append({
                'original_detection': detection,
                'prediction': 'target' if prediction == 1 else 'clutter',
                'confidence': confidence,
                'details': details,
                'kept': prediction == 1 and confidence >= min_confidence
            })
        
        # Create filtered frame data
        filtered_frame = frame_data.copy()
        filtered_frame['filtered_detections'] = filtered_detections
        filtered_frame['classification_metadata'] = classification_metadata
        filtered_frame['filter_stats'] = {
            'original_count': len(all_detections),
            'filtered_count': len(filtered_detections),
            'removed_count': len(all_detections) - len(filtered_detections),
            'removal_rate': 1 - (len(filtered_detections) / len(all_detections)) if all_detections else 0
        }
        
        return filtered_frame
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold."""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
        print(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def get_classification_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Get a summary of classification results for a list of detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            summary: Classification summary statistics
        """
        if not detections:
            return {'total': 0, 'targets': 0, 'clutter': 0, 'target_ratio': 0.0}
        
        results = self.classify_detections_batch(detections)
        
        total = len(results)
        targets = sum(1 for pred, _, _ in results if pred == 1)
        clutter = total - targets
        
        avg_target_confidence = np.mean([conf for pred, conf, _ in results if pred == 1]) if targets > 0 else 0.0
        avg_clutter_confidence = np.mean([conf for pred, conf, _ in results if pred == 0]) if clutter > 0 else 0.0
        
        summary = {
            'total_detections': total,
            'predicted_targets': targets,
            'predicted_clutter': clutter,
            'target_ratio': targets / total if total > 0 else 0.0,
            'avg_target_confidence': avg_target_confidence,
            'avg_clutter_confidence': avg_clutter_confidence,
            'threshold_used': self.confidence_threshold
        }
        
        return summary


def demo_binary_classification():
    """Demonstrate binary classification on existing data."""
    print("Binary Classification Demo")
    print("=" * 30)
    
    # Initialize classifier
    try:
        classifier = BinaryClutterClassifier(
            confidence_threshold=0.7
        )
    except Exception as e:
        print(f"Error loading classifier: {e}")
        print("Please run 'python train_binary_classifier.py' first to train the model.")
        return
    
    # Load some test data
    try:
        from load_data import load_simulation_data
        frames = load_simulation_data('simple_test')
        
        print(f"Loaded {len(frames)} frames for demo")
        
        # Process a few frames
        for i, frame in enumerate(frames[:5]):
            print(f"\nFrame {i+1}:")
            print(f"  Original detections: {len(frame['all_detections'])}")
            
            # Get classification summary
            summary = classifier.get_classification_summary(frame['all_detections'])
            print(f"  Predicted targets: {summary['predicted_targets']}")
            print(f"  Predicted clutter: {summary['predicted_clutter']}")
            print(f"  Target ratio: {summary['target_ratio']:.2f}")
            
            # Filter frame
            filtered_frame = classifier.filter_targets_from_frame(frame, min_confidence=0.7)
            stats = filtered_frame['filter_stats']
            print(f"  After filtering: {stats['filtered_count']} kept, "
                  f"{stats['removed_count']} removed ({stats['removal_rate']:.1%} removal rate)")
            
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for testing and demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Binary Classification Inference')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration on test data')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for classification')
    parser.add_argument('--model-path', type=str, 
                       default='models/binary_classifier/best_model.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_binary_classification()
    else:
        print("Binary Classification Inference Script")
        print("=" * 40)
        print("Use --demo to run demonstration")
        print("Use --help for more options")
        
        # Just test loading the classifier
        try:
            classifier = BinaryClutterClassifier(
                model_path=args.model_path,
                confidence_threshold=args.threshold
            )
            print("Classifier loaded successfully!")
        except Exception as e:
            print(f"Error loading classifier: {e}")


if __name__ == '__main__':
    main()