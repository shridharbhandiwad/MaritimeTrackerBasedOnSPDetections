"""
DeepSORT-Style Maritime Radar Tracker
====================================

Implementation of DeepSORT adapted for maritime radar tracking with:
- Feature extraction network for radar signatures
- Kalman filter for motion prediction
- Hungarian algorithm for association
- Track management with confirmation/deletion logic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Tuple, Optional
import warnings


class RadarFeatureExtractor(nn.Module):
    """Feature extraction network for radar signatures."""
    
    def __init__(self, 
                 input_dim: int = 8,
                 feature_dim: int = 128,
                 hidden_dims: List[int] = [64, 128, 256, 128]):
        """
        Initialize feature extractor.
        
        Args:
            input_dim: Input feature dimension (range, azimuth, doppler, etc.)
            feature_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Build feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, feature_dim))
        
        self.feature_net = nn.Sequential(*layers)
        
        # Normalization layer
        self.l2_norm = nn.functional.normalize
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from radar detections.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            features: Normalized feature vectors [batch_size, feature_dim]
        """
        features = self.feature_net(x)
        # L2 normalize for cosine similarity
        features = self.l2_norm(features, p=2, dim=1)
        return features


class SeaClutterClassifier(nn.Module):
    """Neural network classifier for sea clutter vs target discrimination."""
    
    def __init__(self, 
                 input_dim: int = 8,
                 hidden_dims: List[int] = [64, 32, 16]):
        """
        Initialize clutter classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Binary classification output
        layers.append(nn.Linear(prev_dim, 2))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify detections as clutter (0) or target (1).
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            logits: Classification logits [batch_size, 2]
        """
        return self.classifier(x)


class MaritimeKalmanFilter:
    """Kalman filter adapted for maritime target tracking."""
    
    def __init__(self, 
                 dt: float = 1.0,
                 process_noise: float = 1.0,
                 measurement_noise: float = 1.0):
        """
        Initialize maritime Kalman filter.
        
        Args:
            dt: Time step
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy, ax, ay]
        # Measurements: [x, y, vx, vy] (position and velocity from Doppler)
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        
        # State transition matrix (constant acceleration model)
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.kf.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0, q*dt**2/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2, 0, q*dt**2/2],
            [q*dt**3/2, 0, q*dt**2, 0, q*dt, 0],
            [0, q*dt**3/2, 0, q*dt**2, 0, q*dt],
            [q*dt**2/2, 0, q*dt, 0, q, 0],
            [0, q*dt**2/2, 0, q*dt, 0, q]
        ])
        
        # Measurement noise covariance
        r = measurement_noise
        self.kf.R = np.eye(4) * r
        
        # Initial state covariance
        self.kf.P *= 1000
        
    def initialize(self, measurement: np.ndarray):
        """Initialize filter with first measurement."""
        # measurement: [x, y, vx, vy]
        self.kf.x = np.array([
            measurement[0], measurement[1],  # position
            measurement[2], measurement[3],  # velocity
            0, 0  # acceleration (unknown initially)
        ])
        
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.kf.predict()
        return self.kf.x[:4]  # Return [x, y, vx, vy]
        
    def update(self, measurement: np.ndarray):
        """Update with measurement."""
        self.kf.update(measurement)
        
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.kf.x[:4]  # [x, y, vx, vy]
        
    def get_covariance(self) -> np.ndarray:
        """Get state covariance."""
        return self.kf.P[:4, :4]  # Position-velocity covariance


class Track:
    """Track object for maritime targets."""
    
    def __init__(self, 
                 track_id: int,
                 detection: np.ndarray,
                 feature: np.ndarray,
                 max_age: int = 30,
                 min_hits: int = 3):
        """
        Initialize track.
        
        Args:
            track_id: Unique track identifier
            detection: Initial detection [x, y, vx, vy, ...]
            feature: Appearance feature vector
            max_age: Maximum age without association
            min_hits: Minimum hits for track confirmation
        """
        self.track_id = track_id
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Track state
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'tentative'  # 'tentative', 'confirmed', 'deleted'
        
        # Initialize Kalman filter
        self.kf = MaritimeKalmanFilter()
        self.kf.initialize(detection[:4])
        
        # Feature history for appearance matching
        self.features = [feature]
        self.max_feature_history = 10
        
        # Track history
        self.history = [detection.copy()]
        self.max_history = 50
        
    def predict(self) -> np.ndarray:
        """Predict next state."""
        return self.kf.predict()
        
    def update(self, detection: np.ndarray, feature: np.ndarray):
        """Update track with new detection."""
        self.kf.update(detection[:4])
        
        # Update feature history
        self.features.append(feature)
        if len(self.features) > self.max_feature_history:
            self.features.pop(0)
            
        # Update track history
        self.history.append(detection.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Update track statistics
        self.hits += 1
        self.time_since_update = 0
        
        # Check for confirmation
        if self.state == 'tentative' and self.hits >= self.min_hits:
            self.state = 'confirmed'
            
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.age += 1
        
        # Check for deletion
        if self.time_since_update > self.max_age:
            self.state = 'deleted'
            
    def get_state(self) -> np.ndarray:
        """Get current track state."""
        return self.kf.get_state()
        
    def get_mean_feature(self) -> np.ndarray:
        """Get mean appearance feature."""
        if len(self.features) == 0:
            return np.zeros(128)  # Default feature dimension
        return np.mean(self.features, axis=0)
        
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == 'confirmed'
        
    def is_deleted(self) -> bool:
        """Check if track should be deleted."""
        return self.state == 'deleted'


class MaritimeDeepSORT:
    """DeepSORT tracker adapted for maritime radar."""
    
    def __init__(self,
                 feature_extractor: RadarFeatureExtractor,
                 clutter_classifier: Optional[SeaClutterClassifier] = None,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.5,
                 clutter_threshold: float = 0.5):
        """
        Initialize DeepSORT tracker.
        
        Args:
            feature_extractor: Feature extraction network
            clutter_classifier: Optional clutter classification network
            max_age: Maximum track age without association
            min_hits: Minimum hits for track confirmation
            iou_threshold: IoU threshold for association
            feature_threshold: Feature similarity threshold
            clutter_threshold: Clutter classification threshold
        """
        self.feature_extractor = feature_extractor
        self.clutter_classifier = clutter_classifier
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.clutter_threshold = clutter_threshold
        
        # Track management
        self.tracks = []
        self.next_track_id = 1
        
        # Set networks to eval mode
        self.feature_extractor.eval()
        if self.clutter_classifier is not None:
            self.clutter_classifier.eval()
            
    def _extract_features(self, detections: np.ndarray) -> np.ndarray:
        """Extract features from detections."""
        if len(detections) == 0:
            return np.empty((0, self.feature_extractor.feature_dim))
            
        with torch.no_grad():
            det_tensor = torch.FloatTensor(detections)
            features = self.feature_extractor(det_tensor)
            return features.cpu().numpy()
    
    def _classify_clutter(self, detections: np.ndarray) -> np.ndarray:
        """Classify detections as clutter or targets."""
        if self.clutter_classifier is None or len(detections) == 0:
            return np.ones(len(detections), dtype=bool)  # Assume all are targets
            
        with torch.no_grad():
            det_tensor = torch.FloatTensor(detections)
            logits = self.clutter_classifier(det_tensor)
            probs = F.softmax(logits, dim=1)
            # Return True for targets (class 1)
            return probs[:, 1].cpu().numpy() > self.clutter_threshold
    
    def _calculate_cost_matrix(self, 
                              detections: np.ndarray,
                              features: np.ndarray,
                              tracks: List[Track]) -> np.ndarray:
        """Calculate cost matrix for association."""
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 0))
            
        # Predict track positions
        track_predictions = np.array([track.predict() for track in tracks])
        
        # Calculate position costs (Mahalanobis distance)
        pos_costs = np.zeros((len(detections), len(tracks)))
        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                # Position difference
                pos_diff = detection[:2] - track_predictions[j][:2]
                
                # Covariance matrix
                cov = track.kf.get_covariance()[:2, :2]
                
                # Mahalanobis distance
                try:
                    inv_cov = np.linalg.inv(cov)
                    pos_costs[i, j] = np.sqrt(pos_diff.T @ inv_cov @ pos_diff)
                except np.linalg.LinAlgError:
                    pos_costs[i, j] = np.linalg.norm(pos_diff)
        
        # Calculate feature costs (cosine distance)
        feature_costs = np.zeros((len(detections), len(tracks)))
        for i, feature in enumerate(features):
            for j, track in enumerate(tracks):
                track_feature = track.get_mean_feature()
                # Cosine distance (1 - cosine similarity)
                similarity = np.dot(feature, track_feature) / (
                    np.linalg.norm(feature) * np.linalg.norm(track_feature) + 1e-10)
                feature_costs[i, j] = 1 - similarity
        
        # Combine costs (weighted sum)
        position_weight = 0.7
        feature_weight = 0.3
        total_costs = position_weight * pos_costs + feature_weight * feature_costs
        
        return total_costs
    
    def _associate(self, 
                  detections: np.ndarray,
                  features: np.ndarray,
                  tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using Hungarian algorithm."""
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
            
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections, features, tracks)
        
        # Apply Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter associations based on thresholds
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < self.iou_threshold:  # Using as general association threshold
                matches.append((det_idx, track_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Detection array [n_detections, n_features]
            
        Returns:
            confirmed_tracks: List of confirmed tracks
        """
        # Filter out clutter detections
        if len(detections) > 0:
            target_mask = self._classify_clutter(detections)
            detections = detections[target_mask]
        
        # Extract features
        features = self._extract_features(detections)
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections with tracks
        matches, unmatched_dets, unmatched_tracks = self._associate(
            detections, features, self.tracks)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], features[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                self.next_track_id,
                detections[det_idx],
                features[det_idx],
                self.max_age,
                self.min_hits
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Remove deleted tracks
        self.tracks = [track for track in self.tracks if not track.is_deleted()]
        
        # Return confirmed tracks
        return [track for track in self.tracks if track.is_confirmed()]
    
    def get_track_outputs(self) -> List[Dict]:
        """Get track outputs in standard format."""
        outputs = []
        for track in self.tracks:
            if track.is_confirmed():
                state = track.get_state()
                outputs.append({
                    'track_id': track.track_id,
                    'x': state[0],
                    'y': state[1],
                    'vx': state[2],
                    'vy': state[3],
                    'age': track.age,
                    'hits': track.hits,
                    'state': track.state
                })
        return outputs


def train_feature_extractor(model: RadarFeatureExtractor,
                           train_data: torch.utils.data.DataLoader,
                           val_data: torch.utils.data.DataLoader,
                           num_epochs: int = 100,
                           learning_rate: float = 1e-3) -> Dict:
    """
    Train feature extractor using triplet loss.
    
    Args:
        model: Feature extraction model
        train_data: Training data loader
        val_data: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        training_history: Dictionary with training metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_data:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_features = model(anchor)
            positive_features = model(positive)
            negative_features = model(negative)
            
            loss = triplet_loss(anchor_features, positive_features, negative_features)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_data:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                anchor_features = model(anchor)
                positive_features = model(positive)
                negative_features = model(negative)
                
                loss = triplet_loss(anchor_features, positive_features, negative_features)
                val_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        avg_train_loss = train_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    
    return history