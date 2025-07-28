#!/usr/bin/env python3
"""
Simple Maritime Radar Tracking Example
=====================================

Basic example showing how to use the maritime radar tracking system.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maritime_tracker.data import create_test_scenario
from maritime_tracker.preprocessing import CACFARDetector, CoordinateProcessor
from maritime_tracker.models import RadarFeatureExtractor, MaritimeDeepSORT
from maritime_tracker.evaluation import ComprehensiveEvaluator


def main():
    """Simple example of maritime radar tracking."""
    print("Maritime Radar Tracking - Simple Example")
    print("=" * 45)
    
    # Step 1: Create a test scenario
    print("1. Creating test scenario...")
    simulator = create_test_scenario()
    
    # Step 2: Run simulation for a short duration
    print("2. Running simulation...")
    simulation_data = simulator.run_simulation(duration=60.0, dt=1.0)
    print(f"Generated {len(simulation_data)} frames")
    
    # Step 3: Initialize preprocessing
    print("3. Setting up preprocessing...")
    cfar = CACFARDetector(guard_cells=2, reference_cells=16, pfa=1e-6)
    coord_processor = CoordinateProcessor()
    
    # Step 4: Initialize tracker
    print("4. Setting up tracker...")
    feature_extractor = RadarFeatureExtractor(input_dim=8, feature_dim=128)
    tracker = MaritimeDeepSORT(feature_extractor)
    
    # Step 5: Process simulation data
    print("5. Processing frames...")
    all_tracks = []
    
    for frame_idx, frame_data in enumerate(simulation_data):
        if frame_idx % 10 == 0:
            print(f"   Processing frame {frame_idx+1}/{len(simulation_data)}")
        
        # Extract detections
        detections = frame_data['all_detections']
        
        if not detections:
            continue
        
        # Convert to numpy array
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
        normalized_detections = coord_processor.normalize_coordinates(detection_array)
        
        # Update tracker
        tracks = tracker.update(normalized_detections)
        all_tracks.append(tracks)
    
    # Step 6: Basic evaluation
    print("6. Evaluating performance...")
    evaluator = ComprehensiveEvaluator()
    
    for frame_idx, (frame_data, tracks) in enumerate(zip(simulation_data, all_tracks)):
        # Convert tracks to evaluation format
        predictions = []
        for track in tracks:
            predictions.append({
                'track_id': track['track_id'],
                'x': track['x'],
                'y': track['y'],
                'vx': track.get('vx', 0),
                'vy': track.get('vy', 0)
            })
        
        # Update evaluator
        evaluator.update(frame_data['ground_truth'], predictions, frame_idx)
    
    # Get final metrics
    metrics = evaluator.compute_final_metrics()
    
    # Step 7: Display results
    print("\n7. Results:")
    print("=" * 30)
    print(f"MOTA Score: {metrics['mota']:.3f}")
    print(f"OSPA Distance: {metrics['avg_ospa']:.2f} m")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Track Fragmentation: {metrics['track_fragmentation_rate']:.3f}")
    print(f"Total Tracks Created: {metrics['total_tracks']}")
    
    # Count detections and tracks
    total_detections = sum(len(frame['all_detections']) for frame in simulation_data)
    target_detections = sum(len(frame['target_detections']) for frame in simulation_data)
    clutter_detections = sum(len(frame['clutter_detections']) for frame in simulation_data)
    
    print(f"\nDetection Statistics:")
    print(f"Total Detections: {total_detections}")
    print(f"Target Detections: {target_detections}")
    print(f"Clutter Detections: {clutter_detections}")
    print(f"Clutter Ratio: {clutter_detections/total_detections:.1%}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()