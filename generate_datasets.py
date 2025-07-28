#!/usr/bin/env python3
"""
Maritime Radar Dataset Generator
===============================

Generate comprehensive datasets for maritime radar tracking including:
- Training datasets with various scenarios
- Validation and test datasets
- Different environmental conditions
- Multiple target types and behaviors
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_tracker.data import (
    RadarParameters, EnvironmentParameters, TargetParameters,
    MaritimeRadarSimulator, create_test_scenario
)


def create_training_scenarios():
    """Create diverse training scenarios with different conditions."""
    scenarios = []
    
    # Base radar parameters
    base_radar = RadarParameters(
        carrier_freq=9.4e9,
        max_range=40000,
        range_resolution=15.0,
        azimuth_beamwidth=1.2
    )
    
    # Scenario 1: Calm seas, multiple ships
    env1 = EnvironmentParameters(sea_state=2, wind_speed=8.0, wind_direction=0.0)
    sim1 = MaritimeRadarSimulator(base_radar, env1)
    
    targets1 = [
        TargetParameters((8000, 12000), (10, -5), 200, target_type='large_ship'),
        TargetParameters((-5000, 15000), (7, 3), 180, target_type='large_ship'),
        TargetParameters((20000, -8000), (-15, 12), 120, target_type='ship'),
        TargetParameters((12000, 18000), (-8, -10), 90, target_type='ship'),
        TargetParameters((-15000, -5000), (12, 8), 40, target_type='boat')
    ]
    
    for target in targets1:
        sim1.add_target(target)
    scenarios.append(('calm_seas_multiple_ships', sim1))
    
    # Scenario 2: Rough seas, fewer targets
    env2 = EnvironmentParameters(sea_state=6, wind_speed=25.0, wind_direction=135.0)
    sim2 = MaritimeRadarSimulator(base_radar, env2)
    
    targets2 = [
        TargetParameters((6000, 8000), (12, -8), 250, target_type='large_ship'),
        TargetParameters((-10000, 20000), (5, 2), 60, target_type='boat'),
        TargetParameters((25000, -5000), (-18, 15), 140, target_type='ship')
    ]
    
    for target in targets2:
        sim2.add_target(target)
    scenarios.append(('rough_seas_sparse_targets', sim2))
    
    # Scenario 3: Dense traffic, moderate seas
    env3 = EnvironmentParameters(sea_state=4, wind_speed=15.0, wind_direction=90.0)
    sim3 = MaritimeRadarSimulator(base_radar, env3)
    
    targets3 = [
        TargetParameters((3000, 5000), (8, -4), 180, target_type='ship'),
        TargetParameters((3500, 5200), (9, -3), 170, target_type='ship'),
        TargetParameters((4200, 4800), (7, -5), 160, target_type='ship'),
        TargetParameters((5000, 6000), (6, -2), 30, target_type='boat'),
        TargetParameters((5500, 6200), (8, -1), 35, target_type='boat'),
        TargetParameters((8000, 10000), (15, -10), 220, target_type='large_ship'),
        TargetParameters((-2000, 8000), (12, 5), 40, target_type='boat')
    ]
    
    for target in targets3:
        sim3.add_target(target)
    scenarios.append(('dense_traffic_moderate_seas', sim3))
    
    # Scenario 4: Long range targets
    long_range_radar = RadarParameters(
        carrier_freq=9.4e9,
        max_range=60000,
        range_resolution=20.0,
        azimuth_beamwidth=0.8
    )
    env4 = EnvironmentParameters(sea_state=3, wind_speed=12.0, wind_direction=45.0)
    sim4 = MaritimeRadarSimulator(long_range_radar, env4)
    
    targets4 = [
        TargetParameters((35000, 40000), (10, -12), 300, target_type='large_ship'),
        TargetParameters((-40000, 25000), (8, 8), 280, target_type='large_ship'),
        TargetParameters((45000, -15000), (-20, 18), 150, target_type='ship'),
        TargetParameters((20000, 35000), (-5, -8), 80, target_type='boat')
    ]
    
    for target in targets4:
        sim4.add_target(target)
    scenarios.append(('long_range_tracking', sim4))
    
    # Scenario 5: Maneuvering targets
    env5 = EnvironmentParameters(sea_state=3, wind_speed=10.0, wind_direction=180.0)
    sim5 = MaritimeRadarSimulator(base_radar, env5)
    
    targets5 = [
        TargetParameters((10000, 15000), (12, -8), 180, target_type='ship', maneuver_capability=0.3),
        TargetParameters((-8000, 12000), (15, 5), 200, target_type='ship', maneuver_capability=0.25),
        TargetParameters((18000, -5000), (-10, 15), 50, target_type='boat', maneuver_capability=0.4)
    ]
    
    for target in targets5:
        sim5.add_target(target)
    scenarios.append(('maneuvering_targets', sim5))
    
    return scenarios


def generate_dataset(scenario_name, simulator, duration, dt, output_dir):
    """Generate a single dataset from a scenario."""
    print(f"Generating {scenario_name} dataset...")
    
    # Run simulation
    simulation_data = simulator.run_simulation(duration=duration, dt=dt)
    
    # Create output directory
    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simulation data
    with open(scenario_dir / 'simulation_data.pkl', 'wb') as f:
        pickle.dump(simulation_data, f)
    
    # Save metadata
    metadata = {
        'scenario_name': scenario_name,
        'duration': duration,
        'dt': dt,
        'num_frames': len(simulation_data),
        'radar_params': {
            'carrier_freq': simulator.radar_params.carrier_freq,
            'max_range': simulator.radar_params.max_range,
            'range_resolution': simulator.radar_params.range_resolution,
            'azimuth_beamwidth': simulator.radar_params.azimuth_beamwidth
        },
        'env_params': {
            'sea_state': simulator.env_params.sea_state,
            'wind_speed': simulator.env_params.wind_speed,
            'wind_direction': simulator.env_params.wind_direction
        },
        'num_targets': len(simulator.targets),
        'generated_at': datetime.now().isoformat()
    }
    
    with open(scenario_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate statistics
    total_detections = sum(len(frame['all_detections']) for frame in simulation_data)
    target_detections = sum(len(frame['target_detections']) for frame in simulation_data)
    clutter_detections = sum(len(frame['clutter_detections']) for frame in simulation_data)
    
    stats = {
        'total_detections': total_detections,
        'target_detections': target_detections,
        'clutter_detections': clutter_detections,
        'clutter_ratio': clutter_detections / total_detections if total_detections > 0 else 0,
        'avg_detections_per_frame': total_detections / len(simulation_data),
        'avg_targets_per_frame': target_detections / len(simulation_data)
    }
    
    with open(scenario_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Generated {len(simulation_data)} frames")
    print(f"  Total detections: {total_detections}")
    print(f"  Clutter ratio: {stats['clutter_ratio']:.1%}")
    
    return scenario_dir, stats


def create_simple_test_data():
    """Create simple test data using the built-in test scenario."""
    print("Creating simple test dataset...")
    
    simulator = create_test_scenario()
    output_dir = Path('data') / 'simple_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate short test data
    simulation_data = simulator.run_simulation(duration=120.0, dt=1.0)
    
    # Save data
    with open(output_dir / 'simulation_data.pkl', 'wb') as f:
        pickle.dump(simulation_data, f)
    
    # Save metadata
    metadata = {
        'scenario_name': 'simple_test',
        'duration': 120.0,
        'dt': 1.0,
        'num_frames': len(simulation_data),
        'description': 'Simple test scenario with 3 targets',
        'generated_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Generated {len(simulation_data)} frames")
    return output_dir


def main():
    """Generate comprehensive maritime radar datasets."""
    print("Maritime Radar Dataset Generator")
    print("=" * 50)
    
    # Create output directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate simple test data first
    simple_dir = create_simple_test_data()
    
    # Generate training scenarios
    scenarios = create_training_scenarios()
    
    # Generate datasets
    training_stats = []
    for scenario_name, simulator in scenarios:
        scenario_dir, stats = generate_dataset(
            scenario_name, simulator,
            duration=300.0,  # 5 minutes
            dt=1.0,          # 1 second intervals
            output_dir=data_dir / 'training'
        )
        training_stats.append((scenario_name, stats))
    
    # Generate validation dataset (shorter duration)
    print("\nGenerating validation dataset...")
    val_simulator = create_test_scenario()
    val_dir, val_stats = generate_dataset(
        'validation',
        val_simulator,
        duration=180.0,  # 3 minutes
        dt=1.0,
        output_dir=data_dir
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("Dataset Generation Complete!")
    print("=" * 50)
    
    print(f"\nGenerated datasets in: {data_dir.absolute()}")
    print(f"  - Simple test: {simple_dir.name}")
    print(f"  - Validation: {val_dir.name}")
    
    print(f"\nTraining scenarios:")
    for scenario_name, stats in training_stats:
        print(f"  - {scenario_name}: {stats['total_detections']} detections, "
              f"{stats['clutter_ratio']:.1%} clutter")
    
    print(f"\nValidation: {val_stats['total_detections']} detections, "
          f"{val_stats['clutter_ratio']:.1%} clutter")
    
    # Create dataset index
    dataset_index = {
        'simple_test': {
            'path': str(simple_dir),
            'type': 'test',
            'description': 'Simple test scenario'
        },
        'validation': {
            'path': str(val_dir),
            'type': 'validation',
            'description': 'Validation dataset'
        }
    }
    
    for scenario_name, _ in training_stats:
        dataset_index[scenario_name] = {
            'path': str(data_dir / 'training' / scenario_name),
            'type': 'training',
            'description': f'Training scenario: {scenario_name}'
        }
    
    with open(data_dir / 'dataset_index.json', 'w') as f:
        json.dump(dataset_index, f, indent=2)
    
    print(f"\nDataset index saved to: {data_dir / 'dataset_index.json'}")
    print("\nYou can now use these datasets for training and evaluation!")


if __name__ == '__main__':
    main()