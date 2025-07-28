#!/usr/bin/env python3
"""
Binary Classification Data Preparation
=====================================

Prepare datasets for binary classification of clutter tracks vs actual target tracks.
This script enhances the existing data with better clutter generation and creates
balanced datasets for training the SeaClutterClassifier.
"""

import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_tracker.data.simulator import (
    MaritimeRadarSimulator, RadarParameters, EnvironmentParameters, 
    TargetParameters, SeaClutterModel
)
from load_data import load_dataset_index, load_simulation_data


def enhance_clutter_generation(simulator: MaritimeRadarSimulator, 
                             clutter_multiplier: float = 50.0) -> MaritimeRadarSimulator:
    """
    Enhance clutter generation by increasing false alarm rate.
    
    Args:
        simulator: Original simulator
        clutter_multiplier: Factor to increase clutter generation
        
    Returns:
        Enhanced simulator with more realistic clutter levels
    """
    # Create a copy of the simulator with enhanced clutter generation
    enhanced_sim = MaritimeRadarSimulator(simulator.radar_params, simulator.env_params)
    
    # Copy targets
    for target in simulator.targets:
        enhanced_sim.add_target(target.target_params)
    
    # Monkey patch the clutter generation method to produce more clutter
    original_generate_clutter = enhanced_sim._generate_clutter_detections
    
    def enhanced_generate_clutter(frame_time: float) -> List[Dict]:
        """Enhanced clutter generation with higher false alarm rate."""
        clutter_detections = []
        
        # Much higher base false alarm rate for realistic clutter levels
        base_far = 1e-4 * clutter_multiplier  # Increased base FAR
        
        # Generate clutter in a more distributed manner
        n_clutter_cells = int(np.random.poisson(10 + 5 * enhanced_sim.env_params.sea_state))
        
        for _ in range(n_clutter_cells):
            # Random range and azimuth within coverage
            range_val = np.random.uniform(1000, enhanced_sim.radar_params.max_range * 0.8)
            azimuth = np.random.uniform(-60, 60)  # 120 degree coverage
            
            # Get clutter characteristics based on range and sea state
            clutter_rcs = -40 + 5 * enhanced_sim.env_params.sea_state + np.random.normal(0, 8)
            
            # Clutter Doppler characteristics (mainly zero mean with spread)
            clutter_doppler = np.random.normal(0, 0.5 + 0.3 * enhanced_sim.env_params.sea_state)
            
            # Add range and azimuth noise
            range_noise = np.random.normal(0, enhanced_sim.radar_params.range_resolution / 3)
            azimuth_noise = np.random.normal(0, enhanced_sim.radar_params.azimuth_beamwidth / 3)
            
            detection = {
                'range': range_val + range_noise,
                'azimuth': azimuth + azimuth_noise,
                'doppler': clutter_doppler,
                'rcs': clutter_rcs,
                'snr': clutter_rcs + 15 + np.random.normal(0, 3),
                'target_id': -1,  # Clutter marker
                'frame_time': frame_time,
                'is_clutter': True
            }
            
            # Convert to Cartesian
            x = detection['range'] * np.cos(np.radians(detection['azimuth']))
            y = detection['range'] * np.sin(np.radians(detection['azimuth']))
            
            detection.update({
                'x': x,
                'y': y,
                'vx': np.random.normal(0, 0.2),  # Small velocity noise for clutter
                'vy': np.random.normal(0, 0.2)
            })
            
            clutter_detections.append(detection)
        
        return clutter_detections
    
    # Replace the method
    enhanced_sim._generate_clutter_detections = enhanced_generate_clutter
    
    return enhanced_sim


def extract_detection_features(detection: Dict) -> np.ndarray:
    """
    Extract feature vector from a detection for binary classification.
    
    Args:
        detection: Detection dictionary
        
    Returns:
        features: Feature vector [range, azimuth, doppler, rcs, snr, vx, vy, range_rate]
    """
    # Calculate range rate from velocity
    range_rate = (detection['vx'] * np.cos(np.radians(detection['azimuth'])) + 
                  detection['vy'] * np.sin(np.radians(detection['azimuth'])))
    
    features = np.array([
        detection['range'],
        detection['azimuth'],
        detection['doppler'],
        detection['rcs'],
        detection['snr'],
        detection['vx'],
        detection['vy'],
        range_rate
    ])
    
    return features


def create_binary_classification_dataset(dataset_names: List[str], 
                                        output_dir: str = 'data/binary_classification',
                                        clutter_multiplier: float = 50.0,
                                        enhance_existing: bool = True) -> Dict[str, Any]:
    """
    Create binary classification dataset from existing simulation data.
    
    Args:
        dataset_names: List of dataset names to process
        output_dir: Output directory for processed data
        clutter_multiplier: Factor to increase clutter generation
        enhance_existing: Whether to enhance existing datasets with more clutter
        
    Returns:
        dataset_info: Information about created dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    dataset_stats = {
        'total_detections': 0,
        'target_detections': 0,
        'clutter_detections': 0,
        'datasets_processed': []
    }
    
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        
        # Load original data
        try:
            frames = load_simulation_data(dataset_name)
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue
        
        # Extract features and labels from existing data
        dataset_targets = 0
        dataset_clutter = 0
        
        for frame in frames:
            # Process target detections
            for detection in frame['target_detections']:
                features = extract_detection_features(detection)
                all_features.append(features)
                all_labels.append(1)  # Target = 1
                all_metadata.append({
                    'dataset': dataset_name,
                    'frame_time': frame['frame_time'],
                    'target_id': detection.get('target_id', -1),
                    'detection_type': 'target'
                })
                dataset_targets += 1
            
            # Process existing clutter detections
            for detection in frame['clutter_detections']:
                features = extract_detection_features(detection)
                all_features.append(features)
                all_labels.append(0)  # Clutter = 0
                all_metadata.append({
                    'dataset': dataset_name,
                    'frame_time': frame['frame_time'],
                    'target_id': -1,
                    'detection_type': 'clutter'
                })
                dataset_clutter += 1
        
        # Enhance with additional clutter if requested
        if enhance_existing and dataset_clutter < dataset_targets:
            print(f"  Enhancing {dataset_name} with additional clutter...")
            
            # Generate additional clutter frames
            # We'll create a simplified clutter generator
            additional_clutter_needed = max(dataset_targets - dataset_clutter, dataset_targets)
            clutter_frames_needed = max(1, additional_clutter_needed // 20)  # ~20 clutter per frame
            
            for i in range(clutter_frames_needed):
                synthetic_frame_time = len(frames) + i
                
                # Generate synthetic clutter detections
                n_clutter = np.random.poisson(15)  # Average 15 clutter per frame
                
                for _ in range(n_clutter):
                    # Synthetic clutter detection
                    range_val = np.random.uniform(2000, 25000)
                    azimuth = np.random.uniform(-50, 50)
                    
                    detection = {
                        'range': range_val,
                        'azimuth': azimuth,
                        'doppler': np.random.normal(0, 1.0),
                        'rcs': np.random.normal(-30, 8),
                        'snr': np.random.normal(12, 4),
                        'target_id': -1,
                        'frame_time': synthetic_frame_time,
                        'x': range_val * np.cos(np.radians(azimuth)),
                        'y': range_val * np.sin(np.radians(azimuth)),
                        'vx': np.random.normal(0, 0.3),
                        'vy': np.random.normal(0, 0.3),
                        'is_clutter': True
                    }
                    
                    features = extract_detection_features(detection)
                    all_features.append(features)
                    all_labels.append(0)  # Clutter = 0
                    all_metadata.append({
                        'dataset': dataset_name,
                        'frame_time': synthetic_frame_time,
                        'target_id': -1,
                        'detection_type': 'synthetic_clutter'
                    })
                    dataset_clutter += 1
        
        dataset_stats['datasets_processed'].append({
            'name': dataset_name,
            'targets': dataset_targets,
            'clutter': dataset_clutter,
            'clutter_ratio': dataset_clutter / (dataset_targets + dataset_clutter) if (dataset_targets + dataset_clutter) > 0 else 0
        })
        
        dataset_stats['target_detections'] += dataset_targets
        dataset_stats['clutter_detections'] += dataset_clutter
        
        print(f"  Processed: {dataset_targets} targets, {dataset_clutter} clutter")
    
    dataset_stats['total_detections'] = dataset_stats['target_detections'] + dataset_stats['clutter_detections']
    dataset_stats['overall_clutter_ratio'] = dataset_stats['clutter_detections'] / dataset_stats['total_detections']
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nDataset Summary:")
    print(f"  Total detections: {len(X)}")
    print(f"  Target detections: {np.sum(y)}")
    print(f"  Clutter detections: {len(y) - np.sum(y)}")
    print(f"  Clutter ratio: {1 - np.mean(y):.3f}")
    
    # Feature names
    feature_names = ['range', 'azimuth', 'doppler', 'rcs', 'snr', 'vx', 'vy', 'range_rate']
    
    # Split into train/validation/test
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, all_metadata, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples ({np.mean(y_train):.3f} target ratio)")
    print(f"  Val:   {len(X_val)} samples ({np.mean(y_val):.3f} target ratio)")
    print(f"  Test:  {len(X_test)} samples ({np.mean(y_test):.3f} target ratio)")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save datasets
    dataset_info = {
        'feature_names': feature_names,
        'dataset_stats': dataset_stats,
        'splits': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }
    
    # Save raw data
    with open(output_path / 'train_data.pkl', 'wb') as f:
        pickle.dump({'X': X_train, 'y': y_train, 'metadata': meta_train}, f)
    
    with open(output_path / 'val_data.pkl', 'wb') as f:
        pickle.dump({'X': X_val, 'y': y_val, 'metadata': meta_val}, f)
    
    with open(output_path / 'test_data.pkl', 'wb') as f:
        pickle.dump({'X': X_test, 'y': y_test, 'metadata': meta_test}, f)
    
    # Save scaled data
    with open(output_path / 'train_data_scaled.pkl', 'wb') as f:
        pickle.dump({'X': X_train_scaled, 'y': y_train, 'metadata': meta_train}, f)
    
    with open(output_path / 'val_data_scaled.pkl', 'wb') as f:
        pickle.dump({'X': X_val_scaled, 'y': y_val, 'metadata': meta_val}, f)
    
    with open(output_path / 'test_data_scaled.pkl', 'wb') as f:
        pickle.dump({'X': X_test_scaled, 'y': y_test, 'metadata': meta_test}, f)
    
    # Save scaler and metadata
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create summary visualization
    create_dataset_visualizations(X, y, feature_names, output_path)
    
    print(f"\nBinary classification dataset saved to: {output_path}")
    
    return dataset_info


def create_dataset_visualizations(X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], output_dir: Path):
    """Create visualizations of the binary classification dataset."""
    
    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]
            
            # Plot distributions for targets and clutter
            targets = X[y == 1, i]
            clutter = X[y == 0, i]
            
            ax.hist(targets, bins=50, alpha=0.7, label='Targets', color='blue', density=True)
            ax.hist(clutter, bins=50, alpha=0.7, label='Clutter', color='red', density=True)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(feature_names) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(X.T)
    
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Correlation Matrix')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            plt.text(j, i, f'{correlation_matrix[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(correlation_matrix[i,j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Visualizations saved")


def main():
    """Main function to prepare binary classification data."""
    print("Binary Classification Data Preparation")
    print("=" * 50)
    
    # Load available datasets
    try:
        index = load_dataset_index()
        training_datasets = [name for name, info in index.items() if info['type'] == 'training']
        
        if not training_datasets:
            print("No training datasets found. Please run generate_datasets.py first.")
            return
        
        print(f"Found training datasets: {training_datasets}")
        
        # Create binary classification dataset
        dataset_info = create_binary_classification_dataset(
            dataset_names=training_datasets,
            clutter_multiplier=50.0,
            enhance_existing=True
        )
        
        print("\n" + "=" * 50)
        print("Binary Classification Data Preparation Complete!")
        print("=" * 50)
        
        print(f"\nDataset Statistics:")
        for dataset in dataset_info['dataset_stats']['datasets_processed']:
            print(f"  {dataset['name']}: {dataset['targets']} targets, {dataset['clutter']} clutter "
                  f"({dataset['clutter_ratio']:.1%} clutter)")
        
        print(f"\nOverall: {dataset_info['dataset_stats']['total_detections']} detections, "
              f"{dataset_info['dataset_stats']['overall_clutter_ratio']:.1%} clutter")
        
        print(f"\nNext steps:")
        print(f"  1. Run 'python train_binary_classifier.py' to train the model")
        print(f"  2. Use the trained model for real-time clutter/target classification")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()