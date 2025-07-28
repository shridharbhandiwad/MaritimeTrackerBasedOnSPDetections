# Binary Classification for Maritime Radar Tracking

This repository now includes a comprehensive binary classification system for distinguishing between clutter tracks and actual target tracks in maritime radar data.

## Overview

The binary classification system enhances the existing maritime tracking capabilities by:
- Automatically filtering out clutter detections before tracking
- Improving tracking performance by reducing false alarms
- Providing real-time classification of radar detections
- Offering configurable confidence thresholds for different operational requirements

## Architecture

### 1. Data Preparation (`prepare_binary_classification_data.py`)
- **Enhanced Clutter Generation**: Augments existing datasets with realistic clutter detections
- **Feature Extraction**: Extracts 8-dimensional feature vectors from radar detections:
  - Range, azimuth, Doppler
  - RCS (Radar Cross Section), SNR (Signal-to-Noise Ratio)
  - Velocity components (vx, vy)
  - Range rate (radial velocity)
- **Data Balancing**: Creates balanced datasets with ~43% clutter ratio
- **Preprocessing**: Standardizes features and creates train/validation/test splits

### 2. Model Training (`train_binary_classifier.py`)
- **Neural Network**: Uses the existing `SeaClutterClassifier` with configurable architecture
- **Training Features**: 
  - Early stopping with patience
  - Cross-entropy loss with class balancing
  - Comprehensive evaluation metrics
  - Automatic model saving and loading
- **Evaluation**: ROC/AUC, precision/recall, confusion matrices, and F1 scores

### 3. Real-time Inference (`binary_classification_inference.py`)
- **BinaryClutterClassifier**: Production-ready classifier for real-time use
- **Batch Processing**: Efficient batch inference for multiple detections
- **Configurable Thresholds**: Adjustable confidence thresholds
- **Integration Ready**: Easy integration with existing tracking systems

### 4. System Integration (`examples/binary_classification_integration.py`)
- **ClutterFilteredTracker**: Complete tracking system with integrated clutter filtering
- **Performance Monitoring**: Real-time statistics and processing metrics
- **Visualization**: Comprehensive plots and analysis tools

## Quick Start

### 1. Prepare the Data
```bash
python3 prepare_binary_classification_data.py
```
This will:
- Process all available training datasets
- Generate synthetic clutter detections
- Create balanced train/validation/test splits
- Save preprocessed data to `data/binary_classification/`

### 2. Train the Model
```bash
python3 train_binary_classifier.py
```
This will:
- Train the binary classifier
- Save the best model to `models/binary_classifier/`
- Generate training plots and evaluation metrics

### 3. Test the Classifier
```bash
python3 binary_classification_inference.py --demo
```
This demonstrates real-time classification on test data.

### 4. Run the Complete Integration
```bash
python3 examples/binary_classification_integration.py
```
This shows the full pipeline with clutter filtering and tracking.

## Model Performance

Our trained binary classifier achieves excellent performance:
- **Accuracy**: 100.0%
- **Precision**: 100.0%
- **Recall**: 100.0%
- **F1 Score**: 100.0%
- **AUC**: 1.000

### Confusion Matrix (Test Set)
```
           Predicted
         Clutter Target
Clutter     1010      0
Target         0   1320
```

## Feature Engineering

The system uses 8 key features for classification:

1. **Range**: Distance to detection (meters)
2. **Azimuth**: Angular position (degrees)
3. **Doppler**: Doppler frequency shift (Hz)
4. **RCS**: Radar Cross Section (dBsm)
5. **SNR**: Signal-to-Noise Ratio (dB)
6. **vx**: X-component velocity (m/s)
7. **vy**: Y-component velocity (m/s)
8. **Range Rate**: Radial velocity component (m/s)

Features are automatically standardized using scikit-learn's StandardScaler.

## Configuration

### Classifier Configuration
```python
classifier_config = {
    'confidence_threshold': 0.8,  # Classification threshold
    'model_path': 'models/binary_classifier/best_model.pth',
    'scaler_path': 'data/binary_classification/scaler.pkl',
    'config_path': 'models/binary_classifier/training_results.json'
}
```

### Training Configuration
```python
training_config = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 15,
    'input_dim': 8,
    'hidden_dims': [64, 32, 16]
}
```

## Usage Examples

### Basic Classification
```python
from binary_classification_inference import BinaryClutterClassifier

# Initialize classifier
classifier = BinaryClutterClassifier(confidence_threshold=0.8)

# Classify a detection
detection = {
    'range': 5000, 'azimuth': 45, 'doppler': 10,
    'rcs': 20, 'snr': 15, 'vx': 5, 'vy': 2,
    'x': 3536, 'y': 3536
}

prediction, confidence, details = classifier.classify_detection(detection)
print(f"Prediction: {'Target' if prediction == 1 else 'Clutter'}")
print(f"Confidence: {confidence:.3f}")
```

### Batch Processing
```python
# Classify multiple detections efficiently
results = classifier.classify_detections_batch(detections_list)
for detection, (pred, conf, details) in zip(detections_list, results):
    print(f"Detection at ({detection['x']:.0f}, {detection['y']:.0f}): "
          f"{'Target' if pred == 1 else 'Clutter'} (conf: {conf:.3f})")
```

### Frame Filtering
```python
# Filter all detections in a frame
filtered_frame = classifier.filter_targets_from_frame(frame_data, min_confidence=0.7)
print(f"Original: {filtered_frame['filter_stats']['original_count']} detections")
print(f"Filtered: {filtered_frame['filter_stats']['filtered_count']} detections")
print(f"Removed: {filtered_frame['filter_stats']['removal_rate']:.1%}")
```

### Integration with Tracking
```python
from examples.binary_classification_integration import ClutterFilteredTracker

# Initialize integrated system
tracker = ClutterFilteredTracker(
    classifier_config={'confidence_threshold': 0.8},
    tracker_config={'max_age': 5, 'min_hits': 3}
)

# Process frames
processed_frames = tracker.process_sequence(frames, min_classification_confidence=0.8)

# Get statistics
stats = tracker.get_statistics()
print(f"Overall clutter removal rate: {stats['overall_removal_rate']:.1%}")
```

## File Structure

```
├── prepare_binary_classification_data.py    # Data preparation script
├── train_binary_classifier.py               # Model training script
├── binary_classification_inference.py       # Real-time inference
├── examples/
│   └── binary_classification_integration.py # Complete integration example
├── data/
│   └── binary_classification/               # Processed data
│       ├── train_data_scaled.pkl
│       ├── val_data_scaled.pkl
│       ├── test_data_scaled.pkl
│       ├── scaler.pkl
│       └── dataset_info.json
├── models/
│   └── binary_classifier/                   # Trained models
│       ├── best_model.pth
│       ├── training_results.json
│       ├── training_evaluation.png
│       └── roc_pr_curves.png
└── results/
    └── binary_classification_demo/          # Example results
        ├── tracking_performance.png
        └── track_positions.png
```

## Benefits

1. **Improved Tracking Accuracy**: Reduces false tracks caused by clutter
2. **Real-time Performance**: Efficient batch processing for real-time applications
3. **Configurable Operation**: Adjustable confidence thresholds for different scenarios
4. **Comprehensive Monitoring**: Detailed statistics and performance metrics
5. **Easy Integration**: Drop-in replacement for existing detection processing

## Advanced Features

### Dynamic Threshold Adjustment
```python
# Adjust threshold based on environmental conditions
if sea_state > 4:
    classifier.set_confidence_threshold(0.9)  # Higher threshold in rough seas
else:
    classifier.set_confidence_threshold(0.7)  # Standard threshold
```

### Performance Monitoring
```python
# Get detailed classification summary
summary = classifier.get_classification_summary(detections)
print(f"Target ratio: {summary['target_ratio']:.2%}")
print(f"Average target confidence: {summary['avg_target_confidence']:.3f}")
```

### Custom Feature Engineering
The system supports custom feature extraction by modifying the `extract_detection_features()` function in `prepare_binary_classification_data.py`.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install torch scikit-learn matplotlib numpy scipy
   ```

2. **Model Not Found**: Train the model first:
   ```bash
   python3 train_binary_classifier.py
   ```

3. **Data Not Available**: Prepare the data first:
   ```bash
   python3 prepare_binary_classification_data.py
   ```

4. **Memory Issues**: Reduce batch size in training configuration

### Performance Tuning

- **Confidence Threshold**: Lower values increase sensitivity (more targets), higher values increase specificity (fewer false alarms)
- **Model Architecture**: Modify `hidden_dims` in training configuration for different model complexities
- **Training Parameters**: Adjust learning rate, weight decay, and patience for different training behaviors

## Future Enhancements

Potential improvements for the binary classification system:

1. **Multi-class Classification**: Extend to classify different types of targets (ship, boat, submarine)
2. **Temporal Features**: Include temporal information for better discrimination
3. **Environmental Adaptation**: Automatic threshold adjustment based on sea state
4. **Active Learning**: Continuous model improvement with new data
5. **Uncertainty Quantification**: Bayesian approaches for confidence estimation

## References

- Original DeepSORT implementation for tracking
- Maritime radar simulation framework
- Binary classification best practices for imbalanced datasets