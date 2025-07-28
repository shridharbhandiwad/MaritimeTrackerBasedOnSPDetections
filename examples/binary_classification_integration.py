#!/usr/bin/env python3
"""
Binary Classification Integration Example
=========================================

This example demonstrates how to integrate the binary clutter/target classifier
with the maritime radar tracking system for real-time clutter filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_simulation_data
from binary_classification_inference import BinaryClutterClassifier
from maritime_tracker.models.deep_sort import MaritimeDeepSORT
from maritime_tracker.data.simulator import (
    MaritimeRadarSimulator, RadarParameters, EnvironmentParameters, TargetParameters
)


class ClutterFilteredTracker:
    """Maritime tracker with integrated clutter filtering."""
    
    def __init__(self, 
                 classifier_config: dict = None,
                 tracker_config: dict = None):
        """
        Initialize the filtered tracker.
        
        Args:
            classifier_config: Configuration for the clutter classifier
            tracker_config: Configuration for the tracker
        """
        # Initialize binary classifier
        classifier_config = classifier_config or {}
        self.classifier = BinaryClutterClassifier(
            confidence_threshold=classifier_config.get('confidence_threshold', 0.7),
            model_path=classifier_config.get('model_path', 'models/binary_classifier/best_model.pth'),
            scaler_path=classifier_config.get('scaler_path', 'data/binary_classification/scaler.pkl'),
            config_path=classifier_config.get('config_path', 'models/binary_classifier/training_results.json')
        )
        
        # Initialize tracker
        tracker_config = tracker_config or {}
        
        # Create feature extractor for tracking
        from maritime_tracker.models.deep_sort import RadarFeatureExtractor
        feature_extractor = RadarFeatureExtractor(
            input_dim=9,  # [x, y, vx, vy, range, azimuth, doppler, rcs, snr]
            feature_dim=128
        )
        
        self.tracker = MaritimeDeepSORT(
            feature_extractor=feature_extractor,
            clutter_classifier=None,  # We use our own binary classifier
            max_age=tracker_config.get('max_age', 5),
            min_hits=tracker_config.get('min_hits', 3),
            iou_threshold=tracker_config.get('iou_threshold', 0.3)
        )
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'removed_clutter': 0,
            'frames_processed': 0
        }
        
        print("Clutter-filtered tracker initialized")
    
    def process_frame(self, frame_data: dict, 
                     min_classification_confidence: float = 0.7) -> dict:
        """
        Process a single frame with clutter filtering and tracking.
        
        Args:
            frame_data: Frame data dictionary
            min_classification_confidence: Minimum confidence for keeping detections
            
        Returns:
            processed_frame: Frame data with tracking and filtering results
        """
        original_detections = frame_data.get('all_detections', [])
        
        if not original_detections:
            # No detections to process
            empty_result = {
                'frame_time': frame_data.get('frame_time', 0),
                'original_detections': [],
                'filtered_detections': [],
                'tracks': [],
                'classification_stats': {
                    'original_count': 0,
                    'filtered_count': 0,
                    'removed_count': 0,
                    'removal_rate': 0.0
                }
            }
            return empty_result
        
        # Step 1: Apply clutter filtering
        filtered_frame = self.classifier.filter_targets_from_frame(
            frame_data, 
            min_confidence=min_classification_confidence
        )
        
        filtered_detections = filtered_frame['filtered_detections']
        
        # Step 2: Apply tracking to filtered detections
        # Convert detections to format expected by tracker
        tracking_detections = []
        
        for detection in filtered_detections:
            # Create detection array for tracking
            # Format: [x, y, vx, vy, range, azimuth, doppler, rcs, snr]
            tracking_detection = np.array([
                detection['x'],
                detection['y'],
                detection['vx'],
                detection['vy'],
                detection['range'],
                detection['azimuth'],
                detection['doppler'],
                detection['rcs'],
                detection['snr']
            ])
            tracking_detections.append(tracking_detection)
        
        # Update tracker
        if tracking_detections:
            tracks = self.tracker.update(np.array(tracking_detections))
        else:
            tracks = []
        
        # Update statistics
        self.stats['total_detections'] += len(original_detections)
        self.stats['filtered_detections'] += len(filtered_detections)
        self.stats['removed_clutter'] += (len(original_detections) - len(filtered_detections))
        self.stats['frames_processed'] += 1
        
        # Prepare results
        processed_frame = {
            'frame_time': frame_data.get('frame_time', 0),
            'original_detections': original_detections,
            'filtered_detections': filtered_detections,
            'tracks': [
                {
                    'track_id': track.track_id,
                    'position': track.kf.get_state()[:2].tolist() if hasattr(track, 'kf') else [0, 0],
                    'velocity': track.kf.get_state()[2:4].tolist() if hasattr(track, 'kf') else [0, 0],
                    'time_since_update': track.time_since_update,
                    'hits': track.hits,
                    'age': track.age,
                    'state': track.state
                }
                for track in tracks if track.state != 'deleted'
            ],
            'classification_stats': filtered_frame['filter_stats'],
            'ground_truth': frame_data.get('ground_truth', [])
        }
        
        return processed_frame
    
    def process_sequence(self, frames: list, 
                        min_classification_confidence: float = 0.7) -> list:
        """
        Process a sequence of frames.
        
        Args:
            frames: List of frame data dictionaries
            min_classification_confidence: Minimum confidence for keeping detections
            
        Returns:
            processed_frames: List of processed frame results
        """
        processed_frames = []
        
        print(f"Processing {len(frames)} frames...")
        
        for i, frame in enumerate(frames):
            processed_frame = self.process_frame(frame, min_classification_confidence)
            processed_frames.append(processed_frame)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames")
        
        return processed_frames
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        if self.stats['frames_processed'] == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['avg_detections_per_frame'] = stats['total_detections'] / stats['frames_processed']
        stats['avg_filtered_per_frame'] = stats['filtered_detections'] / stats['frames_processed']
        stats['overall_removal_rate'] = stats['removed_clutter'] / stats['total_detections'] if stats['total_detections'] > 0 else 0
        
        return stats


def visualize_tracking_results(processed_frames: list, output_dir: str = 'results'):
    """Create visualizations of the tracking results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    frame_times = [frame['frame_time'] for frame in processed_frames]
    original_counts = [len(frame['original_detections']) for frame in processed_frames]
    filtered_counts = [len(frame['filtered_detections']) for frame in processed_frames]
    track_counts = [len(frame['tracks']) for frame in processed_frames]
    removal_rates = [frame['classification_stats']['removal_rate'] for frame in processed_frames]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Detection counts over time
    axes[0, 0].plot(frame_times, original_counts, label='Original Detections', linewidth=2)
    axes[0, 0].plot(frame_times, filtered_counts, label='Filtered Detections', linewidth=2)
    axes[0, 0].plot(frame_times, track_counts, label='Active Tracks', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Detection and Track Counts Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clutter removal rate over time
    axes[0, 1].plot(frame_times, removal_rates, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Clutter Removal Rate')
    axes[0, 1].set_title('Clutter Removal Rate Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Detection distribution
    bins = np.arange(0, max(max(original_counts), max(filtered_counts)) + 2) - 0.5
    axes[1, 0].hist(original_counts, bins=bins, alpha=0.7, label='Original', density=True)
    axes[1, 0].hist(filtered_counts, bins=bins, alpha=0.7, label='Filtered', density=True)
    axes[1, 0].set_xlabel('Detections per Frame')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Detection Count Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tracking performance
    active_tracks_over_time = []
    for frame in processed_frames:
        active_track_ids = set()
        for track in frame['tracks']:
            if track['time_since_update'] == 0:  # Recently updated
                active_track_ids.add(track['track_id'])
        active_tracks_over_time.append(len(active_track_ids))
    
    axes[1, 1].plot(frame_times, active_tracks_over_time, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Active Tracks')
    axes[1, 1].set_title('Active Track Count Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'tracking_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Track trajectory plot for last frame
    if processed_frames:
        plt.figure(figsize=(12, 8))
        
        last_frame = processed_frames[-1]
        
        # Plot filtered detections
        if last_frame['filtered_detections']:
            det_x = [d['x'] for d in last_frame['filtered_detections']]
            det_y = [d['y'] for d in last_frame['filtered_detections']]
            plt.scatter(det_x, det_y, c='blue', s=50, alpha=0.6, label='Filtered Detections')
        
        # Plot track positions
        if last_frame['tracks']:
            track_x = [t['position'][0] for t in last_frame['tracks']]
            track_y = [t['position'][1] for t in last_frame['tracks']]
            plt.scatter(track_x, track_y, c='red', s=100, marker='x', linewidth=3, label='Track Positions')
            
            # Add track IDs
            for track in last_frame['tracks']:
                plt.annotate(f"T{track['track_id']}", 
                           (track['position'][0], track['position'][1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        # Plot ground truth if available
        if last_frame.get('ground_truth'):
            gt_x = [gt['x'] for gt in last_frame['ground_truth']]
            gt_y = [gt['y'] for gt in last_frame['ground_truth']]
            plt.scatter(gt_x, gt_y, c='green', s=80, marker='s', alpha=0.7, label='Ground Truth')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Track Positions at Frame {len(processed_frames)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.savefig(output_path / 'track_positions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_path}")


def main():
    """Main demonstration function."""
    print("Binary Classification Integration Example")
    print("=" * 50)
    
    # Configuration
    classifier_config = {
        'confidence_threshold': 0.8,
        'model_path': 'models/binary_classifier/best_model.pth',
        'scaler_path': 'data/binary_classification/scaler.pkl',
        'config_path': 'models/binary_classifier/training_results.json'
    }
    
    tracker_config = {
        'max_age': 5,
        'min_hits': 3,
        'iou_threshold': 0.3
    }
    
    try:
        # Initialize the clutter-filtered tracker
        print("Initializing clutter-filtered tracker...")
        filtered_tracker = ClutterFilteredTracker(
            classifier_config=classifier_config,
            tracker_config=tracker_config
        )
        
        # Load test data
        print("Loading test data...")
        frames = load_simulation_data('simple_test')
        print(f"Loaded {len(frames)} frames")
        
        # Process frames
        print("Processing frames with clutter filtering and tracking...")
        processed_frames = filtered_tracker.process_sequence(
            frames, 
            min_classification_confidence=0.8
        )
        
        # Get statistics
        stats = filtered_tracker.get_statistics()
        
        # Print results
        print("\n" + "=" * 50)
        print("Processing Results")
        print("=" * 50)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Filtered detections: {stats['filtered_detections']}")
        print(f"Removed clutter: {stats['removed_clutter']}")
        print(f"Overall removal rate: {stats['overall_removal_rate']:.1%}")
        print(f"Avg detections per frame: {stats['avg_detections_per_frame']:.1f}")
        print(f"Avg filtered per frame: {stats['avg_filtered_per_frame']:.1f}")
        
        # Track statistics
        unique_tracks = set()
        total_track_updates = 0
        for frame in processed_frames:
            for track in frame['tracks']:
                unique_tracks.add(track['track_id'])
                if track['time_since_update'] == 0:
                    total_track_updates += 1
        
        print(f"\nTracking Results:")
        print(f"Unique tracks created: {len(unique_tracks)}")
        print(f"Total track updates: {total_track_updates}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        visualize_tracking_results(processed_frames, 'results/binary_classification_demo')
        
        # Sample frame analysis
        if processed_frames:
            sample_frame = processed_frames[len(processed_frames)//2]
            print(f"\nSample Frame Analysis (Frame {len(processed_frames)//2}):")
            print(f"  Original detections: {len(sample_frame['original_detections'])}")
            print(f"  Filtered detections: {len(sample_frame['filtered_detections'])}")
            print(f"  Active tracks: {len(sample_frame['tracks'])}")
            print(f"  Removal rate: {sample_frame['classification_stats']['removal_rate']:.1%}")
        
        print("\n" + "=" * 50)
        print("Integration Example Complete!")
        print("=" * 50)
        print("\nKey Benefits of Binary Classification Integration:")
        print("  1. Automatic clutter filtering reduces false tracks")
        print("  2. Improved tracking performance with fewer false alarms")
        print("  3. Real-time processing capability")
        print("  4. Configurable confidence thresholds")
        print("  5. Comprehensive statistics and monitoring")
        
        print(f"\nResults and visualizations saved to: results/binary_classification_demo/")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()