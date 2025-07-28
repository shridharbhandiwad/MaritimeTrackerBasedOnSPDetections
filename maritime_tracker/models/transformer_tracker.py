"""
Transformer-Based Maritime Radar Tracker
=======================================

Implementation of a Transformer-based tracker for maritime radar with:
- Multi-head attention for spatio-temporal feature fusion
- Positional encoding for radar coordinates
- Sequence modeling for track dynamics
- Cross-attention for detection-track association
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Dict, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for radar coordinates and time."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class RadarPositionalEncoding(nn.Module):
    """Radar-specific positional encoding for range, azimuth, and Doppler."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable embeddings for different coordinate systems
        self.range_embedding = nn.Linear(1, d_model // 4)
        self.azimuth_embedding = nn.Linear(1, d_model // 4)
        self.doppler_embedding = nn.Linear(1, d_model // 4)
        self.time_embedding = nn.Linear(1, d_model // 4)
        
    def forward(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detections: [batch_size, seq_len, feature_dim]
                       where features include [range, azimuth, doppler, time, ...]
        """
        batch_size, seq_len, _ = detections.shape
        
        # Extract coordinate features
        range_val = detections[:, :, 0:1]  # Range
        azimuth_val = detections[:, :, 1:2]  # Azimuth
        doppler_val = detections[:, :, 2:3] if detections.shape[-1] > 2 else torch.zeros_like(range_val)
        
        # Create time encoding (sequence position)
        time_val = torch.arange(seq_len, device=detections.device).float()
        time_val = time_val.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        
        # Generate embeddings
        range_emb = self.range_embedding(range_val)
        azimuth_emb = self.azimuth_embedding(azimuth_val)
        doppler_emb = self.doppler_embedding(doppler_val)
        time_emb = self.time_embedding(time_val)
        
        # Concatenate embeddings
        pos_encoding = torch.cat([range_emb, azimuth_emb, doppler_emb, time_emb], dim=-1)
        
        return pos_encoding


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CrossAttentionAssociation(nn.Module):
    """Cross-attention module for detection-track association."""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Association score prediction
        self.association_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, detections: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detections: [batch_size, n_detections, d_model]
            tracks: [batch_size, n_tracks, d_model]
            
        Returns:
            association_matrix: [batch_size, n_detections, n_tracks]
        """
        # Cross-attention: detections attend to tracks
        attended_detections = self.cross_attention(detections, tracks, tracks)
        attended_detections = self.norm(detections + self.dropout(attended_detections))
        
        # Compute association scores
        batch_size, n_detections, d_model = attended_detections.shape
        n_tracks = tracks.shape[1]
        
        # Expand and concatenate for pairwise comparison
        det_expanded = attended_detections.unsqueeze(2).repeat(1, 1, n_tracks, 1)
        track_expanded = tracks.unsqueeze(1).repeat(1, n_detections, 1, 1)
        
        # Element-wise difference and concatenation
        combined = det_expanded * track_expanded  # Element-wise product for similarity
        
        # Flatten for association head
        combined_flat = combined.view(batch_size * n_detections * n_tracks, d_model)
        scores_flat = self.association_head(combined_flat)
        
        # Reshape to association matrix
        association_matrix = scores_flat.view(batch_size, n_detections, n_tracks)
        
        return association_matrix


class TrajectoryPredictor(nn.Module):
    """Trajectory prediction using Transformer."""
    
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4,
                 prediction_horizon: int = 10):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Transformer encoder for history
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Decoder for future prediction
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers//2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 4)  # [x, y, vx, vy]
        
    def forward(self, track_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_history: [batch_size, seq_len, d_model]
            
        Returns:
            predictions: [batch_size, prediction_horizon, 4]
        """
        batch_size, seq_len, _ = track_history.shape
        
        # Encode history
        encoded = track_history
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        # Use last encoded state as initial decoder input
        last_state = encoded[:, -1:, :]  # [batch_size, 1, d_model]
        
        # Predict future states
        predictions = []
        decoder_input = last_state
        
        for _ in range(self.prediction_horizon):
            # Decode next state
            decoded = decoder_input
            for layer in self.decoder_layers:
                decoded = layer(decoded)
            
            # Project to state space
            next_state = self.output_proj(decoded)  # [batch_size, 1, 4]
            predictions.append(next_state)
            
            # Use prediction as next input (teacher forcing during inference)
            # In practice, you might want to add the prediction back to the model space
            decoder_input = decoded
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # [batch_size, horizon, 4]
        
        return predictions


class SeaClutterTransformer(nn.Module):
    """Transformer-based sea clutter classifier."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(8, d_model)  # Project input features
        self.positional_encoding = RadarPositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)  # Binary classification
        )
        
    def forward(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detections: [batch_size, seq_len, input_dim]
            
        Returns:
            logits: [batch_size, seq_len, 2]
        """
        # Project to model dimension
        x = self.input_projection(detections)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(detections)
        x = x + pos_encoding
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Classify each detection
        logits = self.classifier(x)
        
        return logits


class TransformerTracker(nn.Module):
    """Complete Transformer-based maritime radar tracker."""
    
    def __init__(self, 
                 input_dim: int = 8,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 max_detections: int = 100,
                 max_tracks: int = 50,
                 max_sequence_length: int = 50):
        super().__init__()
        
        self.d_model = d_model
        self.max_detections = max_detections
        self.max_tracks = max_tracks
        
        # Input processing
        self.input_projection = nn.Linear(input_dim, d_model)
        self.radar_pos_encoding = RadarPositionalEncoding(d_model)
        
        # Feature extraction
        self.feature_encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_encoder_layers)
        ])
        
        # Association module
        self.association_module = CrossAttentionAssociation(d_model, n_heads)
        
        # Trajectory prediction
        self.trajectory_predictor = TrajectoryPredictor(d_model, n_heads)
        
        # Sea clutter classifier
        self.clutter_classifier = SeaClutterTransformer(d_model, n_heads)
        
        # Track state prediction
        self.state_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 6)  # [x, y, vx, vy, ax, ay]
        )
        
    def encode_detections(self, detections: torch.Tensor) -> torch.Tensor:
        """Encode detection features."""
        # Project to model space
        x = self.input_projection(detections)
        
        # Add positional encoding
        pos_encoding = self.radar_pos_encoding(detections)
        x = x + pos_encoding
        
        # Apply transformer encoder
        for layer in self.feature_encoder:
            x = layer(x)
            
        return x
    
    def classify_clutter(self, detections: torch.Tensor) -> torch.Tensor:
        """Classify sea clutter."""
        return self.clutter_classifier(detections)
    
    def associate_detections(self, detection_features: torch.Tensor,
                           track_features: torch.Tensor) -> torch.Tensor:
        """Associate detections with tracks."""
        return self.association_module(detection_features, track_features)
    
    def predict_trajectory(self, track_history: torch.Tensor) -> torch.Tensor:
        """Predict future trajectory."""
        return self.trajectory_predictor(track_history)
    
    def forward(self, detections: torch.Tensor, 
                track_histories: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete tracker.
        
        Args:
            detections: [batch_size, n_detections, input_dim]
            track_histories: [batch_size, n_tracks, seq_len, input_dim]
            
        Returns:
            outputs: Dictionary containing various outputs
        """
        batch_size = detections.shape[0]
        
        # Encode detections
        detection_features = self.encode_detections(detections)
        
        # Classify clutter
        clutter_logits = self.classify_clutter(detections)
        
        outputs = {
            'detection_features': detection_features,
            'clutter_logits': clutter_logits
        }
        
        # If track histories are provided, perform association and prediction
        if track_histories is not None:
            # Encode track histories (use last state as track feature)
            batch_size_t, n_tracks, seq_len, input_dim = track_histories.shape
            track_hist_flat = track_histories.view(-1, seq_len, input_dim)
            track_features_flat = self.encode_detections(track_hist_flat)
            
            # Use last timestamp as track representation
            track_features = track_features_flat[:, -1, :].view(batch_size_t, n_tracks, self.d_model)
            
            # Association
            association_matrix = self.associate_detections(detection_features, track_features)
            outputs['association_matrix'] = association_matrix
            
            # Trajectory prediction for each track
            trajectory_predictions = []
            for i in range(n_tracks):
                track_hist = track_hist_flat[i::n_tracks]  # Extract i-th track from all batches
                track_encoded = self.encode_detections(track_hist)
                pred = self.predict_trajectory(track_encoded)
                trajectory_predictions.append(pred)
            
            outputs['trajectory_predictions'] = torch.stack(trajectory_predictions, dim=1)
        
        return outputs


class TransformerTrackingPipeline:
    """Complete tracking pipeline using Transformer."""
    
    def __init__(self, model: TransformerTracker, 
                 association_threshold: float = 0.5,
                 clutter_threshold: float = 0.5,
                 max_age: int = 30,
                 min_hits: int = 3):
        self.model = model
        self.association_threshold = association_threshold
        self.clutter_threshold = clutter_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
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
        
        # Convert to tensor
        det_tensor = torch.FloatTensor(detections).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Forward pass
            if len(self.active_tracks) > 0:
                # Prepare track histories
                track_histories = self._prepare_track_histories()
                outputs = self.model(det_tensor, track_histories)
                association_matrix = outputs['association_matrix'][0]  # Remove batch dim
            else:
                outputs = self.model(det_tensor)
                association_matrix = None
            
            # Filter clutter
            clutter_logits = outputs['clutter_logits'][0]  # Remove batch dim
            clutter_probs = F.softmax(clutter_logits, dim=-1)
            target_mask = clutter_probs[:, 1] > self.clutter_threshold
            
            if not target_mask.any():
                return self._age_tracks()
            
            filtered_detections = detections[target_mask.cpu().numpy()]
            
            # Perform association if tracks exist
            if association_matrix is not None and len(self.active_tracks) > 0:
                filtered_association = association_matrix[target_mask]
                matches, unmatched_dets, unmatched_tracks = self._hungarian_assignment(
                    filtered_association)
                
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
    
    def _prepare_track_histories(self) -> torch.Tensor:
        """Prepare track histories for model input."""
        max_history = 20
        histories = []
        
        for track in self.active_tracks:
            # Get recent history
            history = track['history'][-max_history:]
            
            # Pad if necessary
            while len(history) < max_history:
                history = [history[0]] + history
            
            histories.append(history)
        
        # Convert to tensor
        return torch.FloatTensor(histories).unsqueeze(0)  # Add batch dimension
    
    def _hungarian_assignment(self, association_matrix: torch.Tensor) -> Tuple[List, List, List]:
        """Perform Hungarian assignment."""
        from scipy.optimize import linear_sum_assignment
        
        # Convert to cost matrix (1 - association_score)
        cost_matrix = 1 - association_matrix.cpu().numpy()
        
        # Apply Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter based on threshold
        matches = []
        unmatched_detections = list(range(association_matrix.shape[0]))
        unmatched_tracks = list(range(association_matrix.shape[1]))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            score = association_matrix[det_idx, track_idx].item()
            if score > self.association_threshold:
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