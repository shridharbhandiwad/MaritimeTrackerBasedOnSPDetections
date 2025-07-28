# Maritime Radar Data Generation Summary

## 📊 Generated Datasets

Your maritime radar tracking system now has comprehensive datasets ready for training and evaluation. Here's what was created:

### 🗂️ Dataset Overview

**7 Complete Datasets Generated:**

1. **simple_test** (120 frames, 360 detections)
   - Type: Test dataset
   - Targets: 3 ships with simple trajectories
   - Range: 9.4-15.3 km
   - Purpose: Basic testing and validation

2. **validation** (180 frames, 540 detections)
   - Type: Validation dataset
   - Targets: 3 ships
   - Range: 9.4-15.3 km
   - Purpose: Model validation

3. **calm_seas_multiple_ships** (300 frames, 1500 detections)
   - Type: Training dataset
   - Targets: 5 ships in calm conditions
   - Range: 11.7-21.6 km
   - Purpose: Multi-target tracking in ideal conditions

4. **rough_seas_sparse_targets** (300 frames, 900 detections)
   - Type: Training dataset
   - Targets: 3 ships in rough sea conditions
   - Range: 10.0-25.5 km
   - Purpose: Tracking in challenging weather

5. **dense_traffic_moderate_seas** (300 frames, 2100 detections)
   - Type: Training dataset
   - Targets: 7 ships in moderate seas
   - Range: 5.8-14.2 km
   - Purpose: Dense traffic scenarios

6. **long_range_tracking** (300 frames, 1200 detections)
   - Type: Training dataset
   - Targets: 4 ships at long range
   - Range: 38.1-53.2 km
   - Purpose: Long-range detection and tracking

7. **maneuvering_targets** (300 frames, 900 detections)
   - Type: Training dataset
   - Targets: 3 highly maneuvering ships
   - Range: 11.4-18.7 km
   - Purpose: Complex maneuver tracking

### 📁 Data Structure

```
data/
├── dataset_index.json          # Master index of all datasets
├── simple_test/
│   ├── simulation_data.pkl     # Raw simulation data
│   └── metadata.json          # Dataset metadata
├── validation/
│   ├── simulation_data.pkl
│   └── metadata.json
└── training/
    ├── calm_seas_multiple_ships/
    ├── rough_seas_sparse_targets/
    ├── dense_traffic_moderate_seas/
    ├── long_range_tracking/
    └── maneuvering_targets/
```

### 🔧 Available Tools

#### Data Generation
- **`generate_datasets.py`** - Comprehensive dataset generator with multiple scenarios
- **`maritime_tracker/data/simulator.py`** - Core radar simulation engine

#### Data Loading & Analysis
- **`load_data.py`** - Data loading utilities with functions:
  - `load_dataset(name)` - Load complete dataset
  - `list_available_datasets()` - Get list of datasets
  - `load_simulation_data(name)` - Load raw simulation data
  - `print_dataset_summary(name)` - Show dataset statistics

#### Visualization & Demo
- **`demo_data_usage.py`** - Complete demonstration script with:
  - Dataset analysis and statistics
  - Target trajectory visualization
  - Detection scatter plots
  - Range-bearing distributions
  - Time-series analysis

### 📈 Generated Visualizations

- **`data_visualization_simple_test.png`** - 4-panel visualization of simple test scenario
- **`data_visualization_calm_seas_multiple_ships.png`** - 4-panel visualization of training scenario

### 🎯 Data Format

Each frame contains:
```python
frame = {
    'frame_time': 0.0,                    # Timestamp
    'target_detections': [...],           # List of target detections
    'clutter_detections': [...],          # List of clutter detections
    'all_detections': [...],              # All detections combined
    'ground_truth': [...],                # True target positions
    'environment': {...}                  # Environmental parameters
}

# Detection format:
detection = {
    'x': 5000.0,                         # X position (meters)
    'y': 8000.0,                         # Y position (meters)
    'vx': 7.8,                           # X velocity (m/s)
    'vy': -3.0,                          # Y velocity (m/s)
    'range': 9438.0,                     # Range (meters)
    'azimuth': 57.95,                    # Azimuth (degrees)
    'doppler': 99.9,                     # Doppler (Hz)
    'rcs': 23.97,                        # Radar cross section (dBsm)
    'snr': 73.05,                        # Signal-to-noise ratio (dB)
    'target_id': 0,                      # Target identifier
    'frame_time': 0.0                    # Frame timestamp
}
```

## 🚀 Quick Start

### Load and analyze data:
```python
from load_data import load_dataset, list_available_datasets

# List all datasets
datasets = list_available_datasets()
print(datasets)

# Load a specific dataset
data = load_dataset('simple_test')
frames = data['frames']
metadata = data['metadata']
```

### Visualize data:
```python
python3 demo_data_usage.py
```

### Generate new datasets:
```python
python3 generate_datasets.py
```

## 📊 Dataset Statistics Summary

| Dataset | Frames | Detections | Targets | Range (km) | Purpose |
|---------|--------|------------|---------|------------|---------|
| simple_test | 120 | 360 | 3 | 9.4-15.3 | Testing |
| validation | 180 | 540 | 3 | 9.4-15.3 | Validation |
| calm_seas_multiple_ships | 300 | 1,500 | 5 | 11.7-21.6 | Multi-target |
| rough_seas_sparse_targets | 300 | 900 | 3 | 10.0-25.5 | Weather challenge |
| dense_traffic_moderate_seas | 300 | 2,100 | 7 | 5.8-14.2 | Dense traffic |
| long_range_tracking | 300 | 1,200 | 4 | 38.1-53.2 | Long range |
| maneuvering_targets | 300 | 900 | 3 | 11.4-18.7 | Complex maneuvers |

**Total: 1,980 frames, 8,100 radar detections across diverse scenarios**

Your maritime radar tracking system is now ready for development and testing with comprehensive, realistic radar data! 🛟⚓