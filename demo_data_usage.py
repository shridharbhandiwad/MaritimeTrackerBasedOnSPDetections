#!/usr/bin/env python3
"""
Maritime Radar Data Usage Demo
==============================

Demonstrates how to load and visualize the generated maritime radar datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data import load_dataset, load_dataset_index, list_available_datasets


def plot_radar_scenario(dataset_name='simple_test', max_frames=10):
    """Plot a radar scenario showing target tracks and detections."""
    
    print(f"Loading dataset: {dataset_name}")
    try:
        data = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    frames = data['frames']
    num_frames = min(len(frames), max_frames)
    
    print(f"Dataset contains {len(frames)} frames, plotting first {num_frames}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Maritime Radar Dataset: {dataset_name}', fontsize=16)
    
    # Plot 1: Target trajectories over time
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Extract and plot target trajectories
    target_tracks = {}
    for frame_idx, frame in enumerate(frames[:num_frames]):
        if 'ground_truth' in frame:
            for target in frame['ground_truth']:
                target_id = target.get('id', f'target_{len(target_tracks)}')
                if target_id not in target_tracks:
                    target_tracks[target_id] = {'x': [], 'y': []}
                target_tracks[target_id]['x'].append(target['x'])
                target_tracks[target_id]['y'].append(target['y'])
    
    for i, (target_id, track) in enumerate(target_tracks.items()):
        color = colors[i % len(colors)]
        ax1.plot(track['x'], track['y'], 'o-', color=color, label=f'{target_id}', markersize=4)
        # Mark start and end points
        if track['x']:
            ax1.plot(track['x'][0], track['y'][0], 's', color=color, markersize=8, alpha=0.7)
            ax1.plot(track['x'][-1], track['y'][-1], '^', color=color, markersize=8, alpha=0.7)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Target Trajectories')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: Detection scatter for a single frame
    ax2 = axes[0, 1]
    frame_to_show = min(5, len(frames) - 1)
    frame = frames[frame_to_show]
    
    if 'all_detections' in frame:
        detections = frame['all_detections']
        
        # Separate target detections from clutter
        target_x, target_y = [], []
        clutter_x, clutter_y = [], []
        
        # Use the separate lists for target and clutter detections
        for d in frame.get('target_detections', []):
            target_x.append(d['x'])
            target_y.append(d['y'])
        
        for d in frame.get('clutter_detections', []):
            clutter_x.append(d['x'])
            clutter_y.append(d['y'])
        
        if target_x:
            ax2.scatter(target_x, target_y, c='red', s=50, alpha=0.7, label='Target detections')
        if clutter_x:
            ax2.scatter(clutter_x, clutter_y, c='gray', s=20, alpha=0.5, label='Clutter')
    
    # Overlay ground truth positions for this frame
    if 'ground_truth' in frame:
        gt_x = [t['x'] for t in frame['ground_truth']]
        gt_y = [t['y'] for t in frame['ground_truth']]
        ax2.scatter(gt_x, gt_y, c='blue', s=100, marker='x', linewidths=3, label='Ground truth')
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title(f'Detections at Frame {frame_to_show}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # Plot 3: Number of detections over time
    ax3 = axes[1, 0]
    frame_numbers = list(range(len(frames)))
    detection_counts = [len(frame.get('all_detections', [])) for frame in frames]
    target_counts = [len(frame.get('ground_truth', [])) for frame in frames]
    
    ax3.plot(frame_numbers, detection_counts, 'b-', label='Total detections', linewidth=2)
    ax3.plot(frame_numbers, target_counts, 'r--', label='Ground truth targets', linewidth=2)
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Count')
    ax3.set_title('Detections Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Range and bearing distribution
    ax4 = axes[1, 1]
    all_ranges = []
    all_bearings = []
    
    for frame in frames[:num_frames]:
        if 'all_detections' in frame:
            for det in frame['all_detections']:
                x, y = det['x'], det['y']
                range_val = np.sqrt(x**2 + y**2)
                bearing = np.arctan2(y, x) * 180 / np.pi
                all_ranges.append(range_val)
                all_bearings.append(bearing)
    
    if all_ranges:
        ax4.hist2d(all_bearings, all_ranges, bins=20, alpha=0.7)
        ax4.set_xlabel('Bearing (degrees)')
        ax4.set_ylabel('Range (m)')
        ax4.set_title('Detection Range-Bearing Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f'data_visualization_{dataset_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()


def analyze_datasets():
    """Analyze all generated datasets and provide statistics."""
    print("Dataset Analysis")
    print("=" * 50)
    
    try:
        datasets = list_available_datasets()
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return
    
    for dataset_name in datasets:
        print(f"\nAnalyzing {dataset_name}...")
        try:
            data = load_dataset(dataset_name)
            frames = data['frames']
            
            # Basic statistics
            num_frames = len(frames)
            total_detections = sum(len(frame.get('all_detections', [])) for frame in frames)
            total_targets = sum(len(frame.get('ground_truth', [])) for frame in frames)
            
            # Detection statistics
            clutter_count = sum(len(frame.get('clutter_detections', [])) for frame in frames)
            target_detection_count = sum(len(frame.get('target_detections', [])) for frame in frames)
            
            clutter_ratio = (clutter_count / total_detections * 100) if total_detections > 0 else 0
            
            print(f"  Frames: {num_frames}")
            print(f"  Total detections: {total_detections}")
            print(f"  Total ground truth targets: {total_targets}")
            print(f"  Target detections: {target_detection_count}")
            print(f"  Clutter detections: {clutter_count}")
            print(f"  Clutter ratio: {clutter_ratio:.1f}%")
            
            # Range statistics
            ranges = []
            for frame in frames:
                for det in frame.get('all_detections', []):
                    x, y = det['x'], det['y']
                    ranges.append(np.sqrt(x**2 + y**2))
            
            if ranges:
                print(f"  Range stats: min={min(ranges):.1f}m, max={max(ranges):.1f}m, mean={np.mean(ranges):.1f}m")
            
        except Exception as e:
            print(f"  Error analyzing dataset: {e}")


def main():
    """Main function to demonstrate data usage."""
    print("Maritime Radar Data Usage Demo")
    print("=" * 40)
    
    # List available datasets
    print("\nAvailable datasets:")
    try:
        datasets = list_available_datasets()
        for dataset in datasets:
            print(f"  - {dataset}")
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return
    
    # Analyze all datasets
    analyze_datasets()
    
    # Create visualizations for a few key datasets
    print("\nCreating visualizations...")
    
    # Simple test dataset
    if 'simple_test' in datasets:
        plot_radar_scenario('simple_test', max_frames=20)
    
    # One of the training datasets
    training_datasets = [d for d in datasets if 'calm_seas' in d or 'dense_traffic' in d]
    if training_datasets:
        plot_radar_scenario(training_datasets[0], max_frames=15)
    
    print("\nDemo complete! Check the generated visualization files.")


if __name__ == "__main__":
    main()