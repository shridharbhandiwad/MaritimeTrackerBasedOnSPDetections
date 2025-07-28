"""
Maritime Radar Data Module
=========================

Data simulation, loading, and preprocessing utilities for maritime radar tracking.
"""

from .simulator import (
    RadarParameters, EnvironmentParameters, TargetParameters,
    SeaClutterModel, TargetMotionModel, MaritimeRadarSimulator,
    create_test_scenario
)

from .dataset_utils import (
    MaritimeRadarDataset, IPIXDatasetLoader, CSIRDatasetLoader, RASEXDatasetLoader,
    DataAugmentation, TrackingDatasetBuilder, create_train_val_split, create_dataloader
)

__all__ = [
    # Simulation components
    'RadarParameters', 'EnvironmentParameters', 'TargetParameters',
    'SeaClutterModel', 'TargetMotionModel', 'MaritimeRadarSimulator',
    'create_test_scenario',
    
    # Dataset utilities
    'MaritimeRadarDataset', 'IPIXDatasetLoader', 'CSIRDatasetLoader', 'RASEXDatasetLoader',
    'DataAugmentation', 'TrackingDatasetBuilder', 'create_train_val_split', 'create_dataloader'
]