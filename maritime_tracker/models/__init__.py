"""
Maritime Radar Deep Learning Models
==================================

Deep learning models for maritime radar tracking including:
- DeepSORT-style tracking with feature extraction
- Transformer-based tracking with attention mechanisms  
- Graph Neural Networks for spatial-temporal reasoning
"""

from .deep_sort import (
    RadarFeatureExtractor, SeaClutterClassifier, MaritimeKalmanFilter,
    Track, MaritimeDeepSORT, train_feature_extractor
)

from .transformer_tracker import (
    PositionalEncoding, RadarPositionalEncoding, MultiHeadAttention,
    TransformerBlock, CrossAttentionAssociation, TrajectoryPredictor,
    SeaClutterTransformer, TransformerTracker, TransformerTrackingPipeline
)

from .gnn_tracker import (
    RadarGraphConv, SpatialGraphNetwork, TemporalGraphNetwork,
    ClutterClassificationGNN, MaritimeGNNTracker, GNNTrackingPipeline,
    create_spatial_graph, create_temporal_graph
)

__all__ = [
    # DeepSORT components
    'RadarFeatureExtractor', 'SeaClutterClassifier', 'MaritimeKalmanFilter',
    'Track', 'MaritimeDeepSORT', 'train_feature_extractor',
    
    # Transformer components
    'PositionalEncoding', 'RadarPositionalEncoding', 'MultiHeadAttention',
    'TransformerBlock', 'CrossAttentionAssociation', 'TrajectoryPredictor',
    'SeaClutterTransformer', 'TransformerTracker', 'TransformerTrackingPipeline',
    
    # GNN components
    'RadarGraphConv', 'SpatialGraphNetwork', 'TemporalGraphNetwork',
    'ClutterClassificationGNN', 'MaritimeGNNTracker', 'GNNTrackingPipeline',
    'create_spatial_graph', 'create_temporal_graph'
]