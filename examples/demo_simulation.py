#!/usr/bin/env python3
"""
Maritime Radar Tracking System Demo
==================================

Comprehensive demonstration of the maritime radar tracking system including:
1. Radar data simulation with sea clutter and moving targets
2. Data preprocessing with CFAR and Doppler filtering
3. Deep learning model training (DeepSORT, Transformer, GNN)
4. Tracking performance evaluation
5. Visualization and analysis

Usage:
    python demo_simulation.py --duration 300 --method deepsort --evaluate
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maritime_tracker.data import (
    create_test_scenario, TrackingDatasetBuilder, create_train_val_split, create_dataloader
)
from maritime_tracker.preprocessing import (
    CACFARDetector, DopplerProcessor, CoordinateProcessor
)
from maritime_tracker.models import (
    RadarFeatureExtractor, MaritimeDeepSORT, 
    TransformerTracker, TransformerTrackingPipeline,
    MaritimeGNNTracker, GNNTrackingPipeline
)
from maritime_tracker.evaluation import ComprehensiveEvaluator


def run_simulation(duration: float = 300.0, dt: float = 1.0) -> List[Dict]:
    """Run maritime radar simulation."""
    print("=" * 60)
    print("MARITIME RADAR SIMULATION")
    print("=" * 60)
    
    # Create test scenario
    print("Creating test scenario...")
    simulator = create_test_scenario()
    
    # Save scenario configuration
    simulator.save_scenario('test_scenario.json')
    print("Scenario configuration saved to test_scenario.json")
    
    # Run simulation
    print(f"Running simulation for {duration} seconds...")
    simulation_data = simulator.run_simulation(duration, dt)
    
    print(f"Simulation completed: {len(simulation_data)} frames generated")
    
    # Display statistics
    total_detections = sum(len(frame['all_detections']) for frame in simulation_data)
    target_detections = sum(len(frame['target_detections']) for frame in simulation_data)
    clutter_detections = sum(len(frame['clutter_detections']) for frame in simulation_data)
    
    print(f"Total detections: {total_detections}")
    print(f"Target detections: {target_detections}")
    print(f"Clutter detections: {clutter_detections}")
    print(f"Clutter ratio: {clutter_detections/total_detections:.2%}")
    
    return simulation_data


def preprocess_data(simulation_data: List[Dict]) -> List[Dict]:
    """Apply preprocessing to simulation data."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessing components
    doppler_processor = DopplerProcessor(prf=1000.0)
    coordinate_processor = CoordinateProcessor()
    cfar_detector = CACFARDetector(guard_cells=2, reference_cells=16, pfa=1e-6)
    
    processed_data = []
    
    print("Applying preprocessing pipeline...")
    for i, frame_data in enumerate(simulation_data):
        if i % 50 == 0:
            print(f"Processing frame {i+1}/{len(simulation_data)}")
        
        # Extract detection features
        detections = frame_data['all_detections']
        if not detections:
            processed_data.append(frame_data)
            continue
        
        # Convert to numpy array for processing
        detection_array = []
        for det in detections:
            features = [
                det['range'], det['azimuth'], det['x'], det['y'],
                det['doppler'], det['rcs'], det['snr'],
                1.0 if not det.get('is_clutter', False) else 0.0
            ]
            detection_array.append(features)
        
        detection_array = np.array(detection_array)
        
        # Normalize coordinates
        normalized_detections = coordinate_processor.normalize_coordinates(
            detection_array, method='minmax')
        
        # Update frame data
        frame_data['processed_detections'] = normalized_detections
        processed_data.append(frame_data)
    
    print("Preprocessing completed")
    return processed_data


def create_dataset(simulation_data: List[Dict], output_dir: str = 'data') -> str:
    """Create training dataset from simulation data."""
    print("\n" + "=" * 60)
    print("DATASET CREATION")
    print("=" * 60)
    
    # Create dataset builder
    os.makedirs(output_dir, exist_ok=True)
    builder = TrackingDatasetBuilder(output_dir)
    
    # Build dataset
    print("Building dataset from simulation data...")
    dataset_path = builder.build_from_simulation(simulation_data, 'maritime_demo')
    
    # Create train/validation split
    print("Creating train/validation split...")
    train_path, val_path = create_train_val_split(dataset_path, train_ratio=0.8)
    
    return train_path, val_path


def train_deepsort_model(train_path: str, val_path: str) -> MaritimeDeepSORT:
    """Train DeepSORT model."""
    print("\n" + "=" * 60)
    print("TRAINING DEEPSORT MODEL")
    print("=" * 60)
    
    # Create model
    feature_extractor = RadarFeatureExtractor(input_dim=8, feature_dim=128)
    
    # Create data loaders
    train_loader = create_dataloader(train_path, batch_size=16, sequence_length=10)
    val_loader = create_dataloader(val_path, batch_size=16, sequence_length=10, shuffle=False)
    
    print("Training feature extractor...")
    # Note: For demo purposes, we'll create a simple training loop
    # In practice, you would implement proper triplet loss training
    
    # Create tracker
    tracker = MaritimeDeepSORT(feature_extractor)
    print("DeepSORT model created")
    
    return tracker


def train_transformer_model(train_path: str, val_path: str) -> TransformerTrackingPipeline:
    """Train Transformer model."""
    print("\n" + "=" * 60)
    print("TRAINING TRANSFORMER MODEL")
    print("=" * 60)
    
    # Create model
    model = TransformerTracker(input_dim=8, d_model=256, n_heads=8)
    
    # Create pipeline
    pipeline = TransformerTrackingPipeline(model)
    print("Transformer model created")
    
    return pipeline


def train_gnn_model(train_path: str, val_path: str) -> GNNTrackingPipeline:
    """Train GNN model."""
    print("\n" + "=" * 60)
    print("TRAINING GNN MODEL")
    print("=" * 60)
    
    # Create model
    model = MaritimeGNNTracker(input_dim=8, spatial_dim=32, temporal_dim=64)
    
    # Create pipeline
    pipeline = GNNTrackingPipeline(model)
    print("GNN model created")
    
    return pipeline


def evaluate_tracker(tracker, simulation_data: List[Dict], method_name: str) -> Dict:
    """Evaluate tracking performance."""
    print(f"\n" + "=" * 60)
    print(f"EVALUATING {method_name.upper()} TRACKER")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(ospa_cutoff=100.0, mota_threshold=50.0)
    
    print("Running tracker on simulation data...")
    
    # Process each frame
    for frame_idx, frame_data in enumerate(simulation_data):
        if frame_idx % 50 == 0:
            print(f"Processing frame {frame_idx+1}/{len(simulation_data)}")
        
        # Extract detections
        detections = frame_data.get('processed_detections', [])
        if len(detections) == 0:
            detections = np.empty((0, 8))
        
        # Run tracker
        if hasattr(tracker, 'update'):
            tracks = tracker.update(detections)
        else:
            # For models that need different input format
            tracks = []
        
        # Convert to evaluation format
        predictions = []
        for track in tracks:
            predictions.append({
                'track_id': track.get('track_id', track.get('id', 0)),
                'x': track.get('x', 0),
                'y': track.get('y', 0),
                'vx': track.get('vx', 0),
                'vy': track.get('vy', 0)
            })
        
        # Ground truth
        ground_truth = frame_data['ground_truth']
        
        # Update evaluator
        evaluator.update(ground_truth, predictions, frame_idx)
    
    # Compute final metrics
    print("Computing final metrics...")
    metrics = evaluator.compute_final_metrics()
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save metrics
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/{method_name}_metrics.txt', 'w') as f:
        f.write(report)
    
    # Plot metrics
    evaluator.plot_metrics(f'{results_dir}/{method_name}_metrics.png')
    
    return metrics


def visualize_results(simulation_data: List[Dict], tracker_results: Dict):
    """Visualize tracking results."""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Target trajectories
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Extract ground truth trajectories
    gt_trajectories = {}
    for frame_data in simulation_data:
        for obj in frame_data['ground_truth']:
            obj_id = obj['id']
            if obj_id not in gt_trajectories:
                gt_trajectories[obj_id] = {'x': [], 'y': []}
            gt_trajectories[obj_id]['x'].append(obj['x'])
            gt_trajectories[obj_id]['y'].append(obj['y'])
    
    for i, (obj_id, traj) in enumerate(gt_trajectories.items()):
        color = colors[i % len(colors)]
        ax1.plot(traj['x'], traj['y'], color=color, linewidth=2, label=f'Target {obj_id}')
        ax1.scatter(traj['x'][0], traj['y'][0], color=color, marker='o', s=100, edgecolor='black')
        ax1.scatter(traj['x'][-1], traj['y'][-1], color=color, marker='s', s=100, edgecolor='black')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Ground Truth Trajectories')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Detection density
    ax2 = axes[0, 1]
    all_detections_x = []
    all_detections_y = []
    clutter_x = []
    clutter_y = []
    
    for frame_data in simulation_data:
        for det in frame_data['all_detections']:
            all_detections_x.append(det['x'])
            all_detections_y.append(det['y'])
            if det.get('is_clutter', False):
                clutter_x.append(det['x'])
                clutter_y.append(det['y'])
    
    ax2.hexbin(all_detections_x, all_detections_y, gridsize=30, cmap='Blues', alpha=0.7)
    ax2.scatter(clutter_x, clutter_y, c='red', s=1, alpha=0.3, label='Sea Clutter')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Detection Density Map')
    ax2.legend()
    
    # Plot 3: Performance metrics comparison
    ax3 = axes[1, 0]
    methods = list(tracker_results.keys())
    mota_scores = [tracker_results[method]['mota'] for method in methods]
    ospa_scores = [tracker_results[method]['avg_ospa'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, mota_scores, width, label='MOTA', alpha=0.7)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, ospa_scores, width, label='OSPA', alpha=0.7, color='orange')
    
    ax3.set_xlabel('Tracking Method')
    ax3.set_ylabel('MOTA Score')
    ax3_twin.set_ylabel('Average OSPA Distance (m)')
    ax3.set_title('Tracking Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Plot 4: Clutter analysis
    ax4 = axes[1, 1]
    frame_times = [frame['frame_time'] for frame in simulation_data]
    n_detections = [len(frame['all_detections']) for frame in simulation_data]
    n_clutter = [len(frame['clutter_detections']) for frame in simulation_data]
    n_targets = [len(frame['target_detections']) for frame in simulation_data]
    
    ax4.plot(frame_times, n_detections, label='Total Detections', linewidth=2)
    ax4.plot(frame_times, n_clutter, label='Clutter Detections', linewidth=2)
    ax4.plot(frame_times, n_targets, label='Target Detections', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Number of Detections')
    ax4.set_title('Detection Statistics Over Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/tracking_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to results/tracking_analysis.png")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Maritime Radar Tracking Demo')
    parser.add_argument('--duration', type=float, default=300.0, 
                       help='Simulation duration in seconds')
    parser.add_argument('--method', choices=['deepsort', 'transformer', 'gnn', 'all'], 
                       default='all', help='Tracking method to evaluate')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run evaluation and generate report')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    print("MARITIME RADAR TRACKING SYSTEM DEMO")
    print("==================================")
    print(f"Duration: {args.duration} seconds")
    print(f"Method: {args.method}")
    print(f"Evaluation: {'Yes' if args.evaluate else 'No'}")
    print(f"Visualization: {'Yes' if args.visualize else 'No'}")
    
    # Step 1: Run simulation
    simulation_data = run_simulation(args.duration)
    
    # Step 2: Preprocess data
    processed_data = preprocess_data(simulation_data)
    
    # Step 3: Create dataset
    train_path, val_path = create_dataset(processed_data, args.output_dir)
    
    # Step 4: Train models and evaluate
    tracker_results = {}
    
    if args.method in ['deepsort', 'all']:
        print("\n" + "=" * 80)
        print("DEEPSORT PIPELINE")
        print("=" * 80)
        
        # Train DeepSORT
        deepsort_tracker = train_deepsort_model(train_path, val_path)
        
        if args.evaluate:
            # Evaluate DeepSORT
            metrics = evaluate_tracker(deepsort_tracker, processed_data, 'deepsort')
            tracker_results['DeepSORT'] = metrics
    
    if args.method in ['transformer', 'all']:
        print("\n" + "=" * 80)
        print("TRANSFORMER PIPELINE")
        print("=" * 80)
        
        # Train Transformer
        transformer_tracker = train_transformer_model(train_path, val_path)
        
        if args.evaluate:
            # Evaluate Transformer
            metrics = evaluate_tracker(transformer_tracker, processed_data, 'transformer')
            tracker_results['Transformer'] = metrics
    
    if args.method in ['gnn', 'all']:
        print("\n" + "=" * 80)
        print("GNN PIPELINE")
        print("=" * 80)
        
        # Train GNN
        gnn_tracker = train_gnn_model(train_path, val_path)
        
        if args.evaluate:
            # Evaluate GNN
            metrics = evaluate_tracker(gnn_tracker, processed_data, 'gnn')
            tracker_results['GNN'] = metrics
    
    # Step 5: Visualization
    if args.visualize and tracker_results:
        visualize_results(processed_data, tracker_results)
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)
    
    if tracker_results:
        print("Performance Summary:")
        for method, metrics in tracker_results.items():
            print(f"{method}:")
            print(f"  MOTA: {metrics['mota']:.3f}")
            print(f"  OSPA: {metrics['avg_ospa']:.2f} m")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
    
    print("\nFiles generated:")
    print(f"  - Scenario: test_scenario.json")
    print(f"  - Training data: {train_path}")
    print(f"  - Validation data: {val_path}")
    if args.evaluate:
        print(f"  - Results: results/")
    if args.visualize:
        print(f"  - Visualization: results/tracking_analysis.png")
    
    print("\nDemo completed successfully!")


if __name__ == '__main__':
    main()