"""
Graph Neural Network Models for Maritime Radar Tracking
======================================================

Implementation of Graph Neural Networks for maritime radar tracking:
- Graph Convolutional Networks (GCN) for spatial reasoning
- Graph Attention Networks (GAT) for adaptive feature aggregation
- Temporal Graph Networks for track evolution modeling
- Track-to-clutter classification using graph structure
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx
from typing import List, Dict, Tuple, Optional, Union
import networkx as nx
from scipy.spatial.distance import cdist


class RadarGraphConv(MessagePassing):
    """Custom graph convolution for radar data with range/azimuth awareness."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 edge_dim: int = 3, aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Edge transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Transform node features
        x = self.node_mlp(x)
        
        # Transform edge features
        edge_attr = self.edge_mlp(edge_attr)
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Create messages between connected nodes."""
        # Concatenate source and target node features with edge features
        message = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(message)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features."""
        return aggr_out + x  # Residual connection


class SpatialGraphNetwork(nn.Module):
    """Spatial Graph Network for radar detection relationships."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, 
                 num_layers: int = 3, output_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(RadarGraphConv(hidden_dim, hidden_dim))
            else:
                self.conv_layers.append(RadarGraphConv(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr
            
        Returns:
            node_embeddings: [num_nodes, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index, edge_attr)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class TemporalGraphNetwork(nn.Module):
    """Temporal Graph Network for track evolution modeling."""
    
    def __init__(self, node_dim: int = 32, hidden_dim: int = 64, 
                 num_layers: int = 2, sequence_length: int = 10):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(node_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1)
        
        # Graph attention for cross-time relationships
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # Output layers
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, node_sequences: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_sequences: [batch_size, seq_len, node_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            temporal_features: [batch_size, seq_len, node_dim]
        """
        batch_size, seq_len, node_dim = node_sequences.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(node_sequences)
        
        # Self-attention across time
        if attention_mask is not None:
            # Convert mask for attention (True for valid positions)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        attn_out, _ = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        
        # Residual connection and output
        combined = lstm_out + attn_out
        output = self.output_mlp(combined)
        
        return output


class ClutterClassificationGNN(nn.Module):
    """GNN for classifying tracks as clutter or target."""
    
    def __init__(self, node_dim: int = 32, hidden_dim: int = 64, 
                 num_classes: int = 2, num_layers: int = 3):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Graph convolutions
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention mechanism for important features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            logits: [num_nodes, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions with residual connections
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new  # Residual connection
            else:
                x = x_new
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class MaritimeGNNTracker(nn.Module):
    """Complete GNN-based maritime radar tracker."""
    
    def __init__(self, input_dim: int = 8, spatial_dim: int = 32, 
                 temporal_dim: int = 64, num_classes: int = 2):
        super().__init__()
        
        self.spatial_gnn = SpatialGraphNetwork(input_dim, 64, 3, spatial_dim)
        self.temporal_gnn = TemporalGraphNetwork(spatial_dim, temporal_dim)
        self.clutter_classifier = ClutterClassificationGNN(spatial_dim, 64, num_classes)
        
        # Track prediction head
        self.track_predictor = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [x, y, vx, vy, ax, ay]
        )
        
    def forward(self, spatial_data: Data, 
                temporal_sequences: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            spatial_data: Current frame spatial graph
            temporal_sequences: Historical node sequences [batch, seq_len, spatial_dim]
            
        Returns:
            outputs: Dictionary with spatial features, clutter logits, predictions
        """
        # Spatial processing
        spatial_features = self.spatial_gnn(spatial_data)
        
        # Clutter classification
        spatial_data.x = spatial_features
        clutter_logits = self.clutter_classifier(spatial_data)
        
        outputs = {
            'spatial_features': spatial_features,
            'clutter_logits': clutter_logits
        }
        
        # Temporal processing if sequences provided
        if temporal_sequences is not None:
            temporal_features = self.temporal_gnn(temporal_sequences)
            outputs['temporal_features'] = temporal_features
            
            # Use latest temporal features for prediction
            latest_features = temporal_features[:, -1, :]
            track_predictions = self.track_predictor(latest_features)
            outputs['track_predictions'] = track_predictions
        
        return outputs


def create_spatial_graph(detections: np.ndarray, 
                        max_range: float = 1000.0,
                        max_neighbors: int = 10) -> Data:
    """
    Create spatial graph from radar detections.
    
    Args:
        detections: Detection array [n_detections, n_features]
        max_range: Maximum connection range (meters)
        max_neighbors: Maximum number of neighbors per node
        
    Returns:
        data: PyTorch Geometric Data object
    """
    n_detections = len(detections)
    if n_detections == 0:
        return Data(x=torch.empty(0, 8), edge_index=torch.empty(2, 0, dtype=torch.long),
                   edge_attr=torch.empty(0, 3))
    
    # Extract positions (x, y from range, azimuth)
    if detections.shape[1] >= 2:
        ranges = detections[:, 0]
        azimuths = detections[:, 1]
        
        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(np.radians(azimuths))
        y_coords = ranges * np.sin(np.radians(azimuths))
        positions = np.column_stack([x_coords, y_coords])
    else:
        # Fallback to first two features as positions
        positions = detections[:, :2]
    
    # Calculate pairwise distances
    distances = cdist(positions, positions)
    
    # Create edges based on distance threshold
    edge_list = []
    edge_attributes = []
    
    for i in range(n_detections):
        # Find neighbors within range
        neighbor_indices = np.where((distances[i] <= max_range) & (distances[i] > 0))[0]
        
        # Limit number of neighbors
        if len(neighbor_indices) > max_neighbors:
            neighbor_indices = neighbor_indices[np.argsort(distances[i][neighbor_indices])][:max_neighbors]
        
        for j in neighbor_indices:
            edge_list.append([i, j])
            
            # Edge attributes: [distance, relative_angle, relative_doppler]
            dist = distances[i, j]
            rel_angle = np.arctan2(positions[j, 1] - positions[i, 1],
                                 positions[j, 0] - positions[i, 0])
            
            # Relative Doppler (if available)
            if detections.shape[1] > 4:  # Assuming Doppler is 5th feature
                rel_doppler = detections[j, 4] - detections[i, 4]
            else:
                rel_doppler = 0.0
            
            edge_attributes.append([dist, rel_angle, rel_doppler])
    
    # Convert to tensors
    x = torch.FloatTensor(detections)
    
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attributes)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, 3)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_temporal_graph(track_histories: List[np.ndarray],
                         max_temporal_range: int = 5) -> List[Data]:
    """
    Create temporal graphs from track histories.
    
    Args:
        track_histories: List of track history arrays
        max_temporal_range: Maximum temporal connection range
        
    Returns:
        temporal_graphs: List of temporal graph data
    """
    temporal_graphs = []
    
    for track_history in track_histories:
        if len(track_history) < 2:
            continue
            
        seq_len = len(track_history)
        
        # Create temporal edges (connect consecutive time steps)
        edge_list = []
        edge_attributes = []
        
        for t in range(seq_len - 1):
            # Connect current time to next time
            edge_list.append([t, t + 1])
            
            # Edge attributes: time difference, velocity, acceleration
            dt = 1.0  # Assuming unit time steps
            
            if track_history.shape[1] >= 4:  # Has velocity info
                vel_change = np.linalg.norm(track_history[t+1, 2:4] - track_history[t, 2:4])
            else:
                vel_change = 0.0
            
            pos_change = np.linalg.norm(track_history[t+1, :2] - track_history[t, :2])
            
            edge_attributes.append([dt, vel_change, pos_change])
            
            # Also connect to previous time steps within range
            for k in range(1, min(max_temporal_range, t + 1)):
                if t - k >= 0:
                    edge_list.append([t, t - k])
                    
                    dt_k = k
                    vel_change_k = vel_change * k if track_history.shape[1] >= 4 else 0.0
                    pos_change_k = np.linalg.norm(track_history[t, :2] - track_history[t-k, :2])
                    
                    edge_attributes.append([dt_k, vel_change_k, pos_change_k])
        
        # Convert to tensors
        x = torch.FloatTensor(track_history)
        
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_attributes)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 3)
        
        temporal_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    
    return temporal_graphs


class GNNTrackingPipeline:
    """Complete GNN-based tracking pipeline."""
    
    def __init__(self, model: MaritimeGNNTracker,
                 clutter_threshold: float = 0.5,
                 max_age: int = 30,
                 min_hits: int = 3,
                 max_range: float = 1000.0):
        self.model = model
        self.clutter_threshold = clutter_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_range = max_range
        
        # Track management
        self.active_tracks = []
        self.next_track_id = 1
        
        self.model.eval()
    
    def update(self, detections: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Detection array [n_detections, n_features]
            
        Returns:
            track_outputs: List of active track dictionaries
        """
        if len(detections) == 0:
            return self._age_tracks()
        
        # Create spatial graph
        spatial_data = create_spatial_graph(detections, self.max_range)
        
        with torch.no_grad():
            # Forward pass
            if len(self.active_tracks) > 0:
                # Prepare temporal sequences
                temporal_sequences = self._prepare_temporal_sequences()
                outputs = self.model(spatial_data, temporal_sequences)
            else:
                outputs = self.model(spatial_data)
            
            # Filter clutter
            clutter_logits = outputs['clutter_logits']
            clutter_probs = F.softmax(clutter_logits, dim=-1)
            target_mask = clutter_probs[:, 1] > self.clutter_threshold
            
            if not target_mask.any():
                return self._age_tracks()
            
            filtered_detections = detections[target_mask.cpu().numpy()]
            spatial_features = outputs['spatial_features'][target_mask]
            
            # Association using spatial features
            if len(self.active_tracks) > 0:
                matches, unmatched_dets, unmatched_tracks = self._associate_detections(
                    filtered_detections, spatial_features)
                
                # Update matched tracks
                self._update_matched_tracks(matches, filtered_detections)
                
                # Age unmatched tracks
                self._age_unmatched_tracks(unmatched_tracks)
                
                # Create new tracks
                self._create_new_tracks(unmatched_dets, filtered_detections)
            else:
                # No existing tracks, create new ones
                self._create_new_tracks(list(range(len(filtered_detections))), 
                                      filtered_detections)
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return self._get_track_outputs()
    
    def _prepare_temporal_sequences(self) -> torch.Tensor:
        """Prepare temporal sequences for active tracks."""
        max_history = 10
        sequences = []
        
        for track in self.active_tracks:
            history = track['history'][-max_history:]
            
            # Pad if necessary
            while len(history) < max_history:
                history = [history[0]] + history
            
            sequences.append(history)
        
        return torch.FloatTensor(sequences)
    
    def _associate_detections(self, detections: np.ndarray, 
                            spatial_features: torch.Tensor) -> Tuple[List, List, List]:
        """Associate detections with tracks using spatial features."""
        if len(self.active_tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Get track features (last spatial features)
        track_features = []
        for track in self.active_tracks:
            if 'last_features' in track:
                track_features.append(track['last_features'])
            else:
                # Fallback: use detection features
                track_features.append(torch.zeros(spatial_features.shape[1]))
        
        track_features = torch.stack(track_features)
        
        # Calculate similarity matrix
        det_features = spatial_features
        similarity_matrix = torch.mm(det_features, track_features.t())
        
        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - similarity_matrix.cpu().numpy()
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter associations
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.active_tracks)))
        
        association_threshold = 0.3
        for det_idx, track_idx in zip(det_indices, track_indices):
            similarity = similarity_matrix[det_idx, track_idx].item()
            if similarity > association_threshold:
                matches.append((det_idx, track_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _update_matched_tracks(self, matches: List, detections: np.ndarray):
        """Update matched tracks."""
        for det_idx, track_idx in matches:
            track = self.active_tracks[track_idx]
            detection = detections[det_idx]
            
            # Update track
            track['history'].append(detection)
            track['hits'] += 1
            track['time_since_update'] = 0
            track['age'] += 1
            
            # Limit history size
            if len(track['history']) > 50:
                track['history'].pop(0)
            
            # Confirm track if enough hits
            if track['hits'] >= self.min_hits:
                track['state'] = 'confirmed'
    
    def _age_unmatched_tracks(self, unmatched_tracks: List):
        """Age unmatched tracks."""
        for track_idx in unmatched_tracks:
            track = self.active_tracks[track_idx]
            track['time_since_update'] += 1
            track['age'] += 1
    
    def _age_tracks(self) -> List[Dict]:
        """Age all tracks when no detections."""
        for track in self.active_tracks:
            track['time_since_update'] += 1
            track['age'] += 1
        
        self._cleanup_tracks()
        return self._get_track_outputs()
    
    def _create_new_tracks(self, detection_indices: List, detections: np.ndarray):
        """Create new tracks."""
        for det_idx in detection_indices:
            detection = detections[det_idx]
            
            new_track = {
                'track_id': self.next_track_id,
                'history': [detection],
                'hits': 1,
                'time_since_update': 0,
                'age': 1,
                'state': 'tentative'
            }
            
            self.active_tracks.append(new_track)
            self.next_track_id += 1
    
    def _cleanup_tracks(self):
        """Remove old tracks."""
        self.active_tracks = [
            track for track in self.active_tracks 
            if track['time_since_update'] <= self.max_age
        ]
    
    def _get_track_outputs(self) -> List[Dict]:
        """Get confirmed track outputs."""
        outputs = []
        for track in self.active_tracks:
            if track['state'] == 'confirmed':
                latest_detection = track['history'][-1]
                outputs.append({
                    'track_id': track['track_id'],
                    'x': latest_detection[0],
                    'y': latest_detection[1],
                    'vx': latest_detection[2] if len(latest_detection) > 2 else 0,
                    'vy': latest_detection[3] if len(latest_detection) > 3 else 0,
                    'age': track['age'],
                    'hits': track['hits'],
                    'state': track['state']
                })
        return outputs