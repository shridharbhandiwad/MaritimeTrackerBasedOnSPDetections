"""
Evaluation Metrics for Maritime Radar Tracking
=============================================

Implementation of comprehensive evaluation metrics for tracking systems:
- OSPA (Optimal SubPattern Assignment) distance
- MOTA/MOTP (Multiple Object Tracking Accuracy/Precision)
- Track fragmentation and continuity metrics
- False track rate and detection metrics
- Maritime-specific evaluation metrics
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional, Union
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class OSPAMetric:
    """Optimal SubPattern Assignment (OSPA) distance metric."""
    
    def __init__(self, cutoff_distance: float = 100.0, order: int = 2):
        """
        Initialize OSPA metric.
        
        Args:
            cutoff_distance: Maximum distance for assignment (meters)
            order: Order parameter (typically 1 or 2)
        """
        self.cutoff_distance = cutoff_distance
        self.order = order
    
    def compute(self, ground_truth: np.ndarray, estimates: np.ndarray) -> Dict[str, float]:
        """
        Compute OSPA distance between ground truth and estimates.
        
        Args:
            ground_truth: GT positions [n_gt, 2] (x, y)
            estimates: Estimated positions [n_est, 2] (x, y)
            
        Returns:
            metrics: Dictionary with OSPA components
        """
        n_gt = len(ground_truth)
        n_est = len(estimates)
        
        if n_gt == 0 and n_est == 0:
            return {'ospa': 0.0, 'localization': 0.0, 'cardinality': 0.0}
        
        if n_gt == 0:
            # Only false alarms
            cardinality_error = self.cutoff_distance
            localization_error = 0.0
        elif n_est == 0:
            # Only missed detections
            cardinality_error = self.cutoff_distance
            localization_error = 0.0
        else:
            # Calculate distance matrix
            distances = cdist(ground_truth, estimates)
            
            # Determine assignment
            m = max(n_gt, n_est)
            
            if n_gt <= n_est:
                # More estimates than GT
                gt_indices, est_indices = linear_sum_assignment(distances)
                assigned_distances = distances[gt_indices, est_indices]
                
                # Clip distances to cutoff
                clipped_distances = np.minimum(assigned_distances, self.cutoff_distance)
                
                # Localization error from assigned pairs
                localization_error = np.mean(clipped_distances ** self.order) ** (1/self.order)
                
                # Cardinality error from extra estimates
                n_extra = n_est - n_gt
                cardinality_component = n_extra * (self.cutoff_distance ** self.order) / m
                
            else:
                # More GT than estimates
                gt_indices, est_indices = linear_sum_assignment(distances)
                assigned_distances = distances[gt_indices, est_indices]
                
                # Clip distances to cutoff
                clipped_distances = np.minimum(assigned_distances, self.cutoff_distance)
                
                # Localization error from assigned pairs
                localization_error = np.mean(clipped_distances ** self.order) ** (1/self.order)
                
                # Cardinality error from missed GT
                n_missed = n_gt - n_est
                cardinality_component = n_missed * (self.cutoff_distance ** self.order) / m
            
            # Total OSPA distance
            total_component = (np.sum(clipped_distances ** self.order) + 
                             cardinality_component) / m
            ospa_distance = total_component ** (1/self.order)
            
            # Separate cardinality error
            cardinality_error = (cardinality_component / m) ** (1/self.order)
        
        return {
            'ospa': ospa_distance if n_gt > 0 or n_est > 0 else 0.0,
            'localization': localization_error,
            'cardinality': cardinality_error
        }


class MOTAMetrics:
    """Multiple Object Tracking Accuracy (MOTA) and Precision (MOTP) metrics."""
    
    def __init__(self, distance_threshold: float = 50.0):
        """
        Initialize MOTA metrics.
        
        Args:
            distance_threshold: Maximum distance for association (meters)
        """
        self.distance_threshold = distance_threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_gt = 0
        self.total_missed = 0
        self.total_false_positives = 0
        self.total_id_switches = 0
        self.total_distance_errors = 0
        self.total_matched = 0
        
        # Track ID mapping for ID switch detection
        self.gt_to_pred_mapping = {}
        self.pred_to_gt_mapping = {}
    
    def update(self, ground_truth: List[Dict], predictions: List[Dict]):
        """
        Update metrics with current frame data.
        
        Args:
            ground_truth: List of GT objects [{'id': int, 'x': float, 'y': float, ...}]
            predictions: List of predicted objects [{'track_id': int, 'x': float, 'y': float, ...}]
        """
        n_gt = len(ground_truth)
        n_pred = len(predictions)
        
        self.total_gt += n_gt
        
        if n_gt == 0 and n_pred == 0:
            return
        
        if n_gt == 0:
            # Only false positives
            self.total_false_positives += n_pred
            return
        
        if n_pred == 0:
            # Only missed detections
            self.total_missed += n_gt
            return
        
        # Extract positions and IDs
        gt_positions = np.array([[obj['x'], obj['y']] for obj in ground_truth])
        pred_positions = np.array([[obj['x'], obj['y']] for obj in predictions])
        
        gt_ids = [obj['id'] for obj in ground_truth]
        pred_ids = [obj['track_id'] for obj in predictions]
        
        # Calculate distance matrix
        distances = cdist(gt_positions, pred_positions)
        
        # Find associations within threshold
        gt_indices, pred_indices = linear_sum_assignment(distances)
        
        # Filter associations by distance threshold
        valid_associations = distances[gt_indices, pred_indices] <= self.distance_threshold
        gt_matched = gt_indices[valid_associations]
        pred_matched = pred_indices[valid_associations]
        
        # Count metrics
        n_matched = len(gt_matched)
        n_missed = n_gt - n_matched
        n_false_positives = n_pred - n_matched
        
        self.total_matched += n_matched
        self.total_missed += n_missed
        self.total_false_positives += n_false_positives
        
        # Calculate distance errors for matched pairs
        if n_matched > 0:
            matched_distances = distances[gt_matched, pred_matched]
            self.total_distance_errors += np.sum(matched_distances)
        
        # Count ID switches
        id_switches = 0
        for i, (gt_idx, pred_idx) in enumerate(zip(gt_matched, pred_matched)):
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]
            
            # Check for ID switches
            if gt_id in self.gt_to_pred_mapping:
                if self.gt_to_pred_mapping[gt_id] != pred_id:
                    id_switches += 1
                    self.gt_to_pred_mapping[gt_id] = pred_id
            else:
                self.gt_to_pred_mapping[gt_id] = pred_id
            
            if pred_id in self.pred_to_gt_mapping:
                if self.pred_to_gt_mapping[pred_id] != gt_id:
                    id_switches += 1
                    self.pred_to_gt_mapping[pred_id] = gt_id
            else:
                self.pred_to_gt_mapping[pred_id] = gt_id
        
        self.total_id_switches += id_switches
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final MOTA/MOTP metrics.
        
        Returns:
            metrics: Dictionary with MOTA, MOTP, and related metrics
        """
        if self.total_gt == 0:
            return {
                'mota': 0.0,
                'motp': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'id_switches': 0,
                'false_positive_rate': 0.0
            }
        
        # MOTA = 1 - (FN + FP + IDSW) / GT
        mota = 1 - (self.total_missed + self.total_false_positives + 
                   self.total_id_switches) / self.total_gt
        
        # MOTP = sum of distance errors / number of matches
        motp = (self.total_distance_errors / self.total_matched 
                if self.total_matched > 0 else float('inf'))
        
        # Additional metrics
        precision = (self.total_matched / 
                    (self.total_matched + self.total_false_positives)
                    if (self.total_matched + self.total_false_positives) > 0 else 0.0)
        
        recall = (self.total_matched / self.total_gt 
                 if self.total_gt > 0 else 0.0)
        
        false_positive_rate = (self.total_false_positives / 
                              (self.total_matched + self.total_false_positives)
                              if (self.total_matched + self.total_false_positives) > 0 else 0.0)
        
        return {
            'mota': mota,
            'motp': motp,
            'precision': precision,
            'recall': recall,
            'id_switches': self.total_id_switches,
            'false_positive_rate': false_positive_rate,
            'total_gt': self.total_gt,
            'total_matched': self.total_matched,
            'total_missed': self.total_missed,
            'total_false_positives': self.total_false_positives
        }


class TrackFragmentationMetrics:
    """Metrics for track fragmentation and continuity analysis."""
    
    def __init__(self):
        """Initialize fragmentation metrics."""
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.track_histories = defaultdict(list)
        self.gt_track_histories = defaultdict(list)
        self.track_fragments = defaultdict(list)
        self.frame_count = 0
    
    def update(self, ground_truth: List[Dict], predictions: List[Dict], frame_id: int):
        """
        Update fragmentation metrics with frame data.
        
        Args:
            ground_truth: List of GT objects
            predictions: List of predicted objects
            frame_id: Current frame identifier
        """
        self.frame_count = max(self.frame_count, frame_id + 1)
        
        # Record GT track histories
        for obj in ground_truth:
            gt_id = obj['id']
            self.gt_track_histories[gt_id].append({
                'frame': frame_id,
                'x': obj['x'],
                'y': obj['y'],
                'active': True
            })
        
        # Record predicted track histories
        for obj in predictions:
            track_id = obj['track_id']
            self.track_histories[track_id].append({
                'frame': frame_id,
                'x': obj['x'],
                'y': obj['y'],
                'active': True
            })
    
    def compute(self) -> Dict[str, float]:
        """
        Compute fragmentation metrics.
        
        Returns:
            metrics: Dictionary with fragmentation metrics
        """
        if not self.track_histories:
            return {
                'avg_track_length': 0.0,
                'track_fragmentation_rate': 0.0,
                'track_completion_rate': 0.0,
                'longest_track_length': 0,
                'total_tracks': 0,
                'avg_gap_length': 0.0
            }
        
        track_lengths = []
        track_gaps = []
        completed_tracks = 0
        
        for track_id, history in self.track_histories.items():
            # Calculate track length
            track_length = len(history)
            track_lengths.append(track_length)
            
            # Check if track spans significant portion of sequence
            first_frame = min(entry['frame'] for entry in history)
            last_frame = max(entry['frame'] for entry in history)
            span = last_frame - first_frame + 1
            
            if span >= 0.8 * self.frame_count:  # Track covers 80% of sequence
                completed_tracks += 1
            
            # Detect gaps in track
            frames = sorted([entry['frame'] for entry in history])
            gaps = []
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i-1] - 1
                if gap > 0:
                    gaps.append(gap)
            track_gaps.extend(gaps)
        
        # Calculate metrics
        avg_track_length = np.mean(track_lengths) if track_lengths else 0.0
        longest_track_length = max(track_lengths) if track_lengths else 0
        total_tracks = len(self.track_histories)
        
        # Fragmentation rate: tracks with gaps / total tracks
        fragmented_tracks = sum(1 for track_id, history in self.track_histories.items()
                              if self._has_gaps(history))
        track_fragmentation_rate = fragmented_tracks / total_tracks if total_tracks > 0 else 0.0
        
        # Completion rate: tracks that span most of sequence / total tracks
        track_completion_rate = completed_tracks / total_tracks if total_tracks > 0 else 0.0
        
        # Average gap length
        avg_gap_length = np.mean(track_gaps) if track_gaps else 0.0
        
        return {
            'avg_track_length': avg_track_length,
            'track_fragmentation_rate': track_fragmentation_rate,
            'track_completion_rate': track_completion_rate,
            'longest_track_length': longest_track_length,
            'total_tracks': total_tracks,
            'avg_gap_length': avg_gap_length,
            'total_gaps': len(track_gaps)
        }
    
    def _has_gaps(self, history: List[Dict]) -> bool:
        """Check if track history has temporal gaps."""
        frames = sorted([entry['frame'] for entry in history])
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:
                return True
        return False


class MaritimeSpecificMetrics:
    """Maritime-specific evaluation metrics."""
    
    def __init__(self, sea_clutter_threshold: float = 0.5):
        """
        Initialize maritime metrics.
        
        Args:
            sea_clutter_threshold: Threshold for sea clutter classification
        """
        self.sea_clutter_threshold = sea_clutter_threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.target_detections = 0
        self.clutter_detections = 0
        self.target_true_positives = 0
        self.target_false_positives = 0
        self.clutter_true_positives = 0
        self.clutter_false_positives = 0
        
        self.velocity_errors = []
        self.range_errors = []
        self.azimuth_errors = []
        
    def update(self, ground_truth: List[Dict], predictions: List[Dict],
               clutter_labels: Optional[np.ndarray] = None,
               clutter_predictions: Optional[np.ndarray] = None):
        """
        Update maritime-specific metrics.
        
        Args:
            ground_truth: List of GT objects with maritime attributes
            predictions: List of predicted objects
            clutter_labels: Ground truth clutter labels [0=target, 1=clutter]
            clutter_predictions: Predicted clutter probabilities
        """
        # Update clutter classification metrics
        if clutter_labels is not None and clutter_predictions is not None:
            clutter_pred_binary = (clutter_predictions > self.sea_clutter_threshold).astype(int)
            
            for true_label, pred_label in zip(clutter_labels, clutter_pred_binary):
                if true_label == 0:  # True target
                    self.target_detections += 1
                    if pred_label == 0:  # Predicted as target
                        self.target_true_positives += 1
                    else:  # Predicted as clutter
                        self.target_false_positives += 1
                else:  # True clutter
                    self.clutter_detections += 1
                    if pred_label == 1:  # Predicted as clutter
                        self.clutter_true_positives += 1
                    else:  # Predicted as target
                        self.clutter_false_positives += 1
        
        # Update tracking accuracy for maritime attributes
        if ground_truth and predictions:
            # Associate GT and predictions (simple nearest neighbor for now)
            gt_positions = np.array([[obj['x'], obj['y']] for obj in ground_truth])
            pred_positions = np.array([[obj['x'], obj['y']] for obj in predictions])
            
            distances = cdist(gt_positions, pred_positions)
            gt_indices, pred_indices = linear_sum_assignment(distances)
            
            # Calculate maritime-specific errors
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                if distances[gt_idx, pred_idx] <= 100.0:  # Valid association
                    gt_obj = ground_truth[gt_idx]
                    pred_obj = predictions[pred_idx]
                    
                    # Velocity error
                    if 'vx' in gt_obj and 'vy' in gt_obj and 'vx' in pred_obj and 'vy' in pred_obj:
                        gt_velocity = np.sqrt(gt_obj['vx']**2 + gt_obj['vy']**2)
                        pred_velocity = np.sqrt(pred_obj['vx']**2 + pred_obj['vy']**2)
                        self.velocity_errors.append(abs(gt_velocity - pred_velocity))
                    
                    # Range error
                    gt_range = np.sqrt(gt_obj['x']**2 + gt_obj['y']**2)
                    pred_range = np.sqrt(pred_obj['x']**2 + pred_obj['y']**2)
                    self.range_errors.append(abs(gt_range - pred_range))
                    
                    # Azimuth error
                    gt_azimuth = np.arctan2(gt_obj['y'], gt_obj['x'])
                    pred_azimuth = np.arctan2(pred_obj['y'], pred_obj['x'])
                    azimuth_diff = abs(gt_azimuth - pred_azimuth)
                    # Handle wrap-around
                    azimuth_diff = min(azimuth_diff, 2*np.pi - azimuth_diff)
                    self.azimuth_errors.append(np.degrees(azimuth_diff))
    
    def compute(self) -> Dict[str, float]:
        """
        Compute maritime-specific metrics.
        
        Returns:
            metrics: Dictionary with maritime metrics
        """
        # Clutter classification metrics
        target_precision = (self.target_true_positives / 
                           (self.target_true_positives + self.clutter_false_positives)
                           if (self.target_true_positives + self.clutter_false_positives) > 0 else 0.0)
        
        target_recall = (self.target_true_positives / self.target_detections
                        if self.target_detections > 0 else 0.0)
        
        clutter_precision = (self.clutter_true_positives / 
                            (self.clutter_true_positives + self.target_false_positives)
                            if (self.clutter_true_positives + self.target_false_positives) > 0 else 0.0)
        
        clutter_recall = (self.clutter_true_positives / self.clutter_detections
                         if self.clutter_detections > 0 else 0.0)
        
        # Maritime tracking accuracy
        avg_velocity_error = np.mean(self.velocity_errors) if self.velocity_errors else 0.0
        avg_range_error = np.mean(self.range_errors) if self.range_errors else 0.0
        avg_azimuth_error = np.mean(self.azimuth_errors) if self.azimuth_errors else 0.0
        
        return {
            'target_precision': target_precision,
            'target_recall': target_recall,
            'clutter_precision': clutter_precision,
            'clutter_recall': clutter_recall,
            'avg_velocity_error_ms': avg_velocity_error,
            'avg_range_error_m': avg_range_error,
            'avg_azimuth_error_deg': avg_azimuth_error,
            'velocity_rmse_ms': np.sqrt(np.mean(np.square(self.velocity_errors))) if self.velocity_errors else 0.0,
            'range_rmse_m': np.sqrt(np.mean(np.square(self.range_errors))) if self.range_errors else 0.0
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""
    
    def __init__(self, ospa_cutoff: float = 100.0, mota_threshold: float = 50.0):
        """
        Initialize comprehensive evaluator.
        
        Args:
            ospa_cutoff: OSPA cutoff distance
            mota_threshold: MOTA association threshold
        """
        self.ospa_metric = OSPAMetric(ospa_cutoff)
        self.mota_metrics = MOTAMetrics(mota_threshold)
        self.fragmentation_metrics = TrackFragmentationMetrics()
        self.maritime_metrics = MaritimeSpecificMetrics()
        
        self.frame_results = []
    
    def reset(self):
        """Reset all metrics."""
        self.mota_metrics.reset()
        self.fragmentation_metrics.reset()
        self.maritime_metrics.reset()
        self.frame_results = []
    
    def update(self, ground_truth: List[Dict], predictions: List[Dict], 
               frame_id: int, **kwargs):
        """
        Update all metrics with frame data.
        
        Args:
            ground_truth: List of GT objects
            predictions: List of predicted objects
            frame_id: Current frame identifier
            **kwargs: Additional data (clutter_labels, clutter_predictions, etc.)
        """
        # Extract positions for OSPA
        if ground_truth and predictions:
            gt_positions = np.array([[obj['x'], obj['y']] for obj in ground_truth])
            pred_positions = np.array([[obj['x'], obj['y']] for obj in predictions])
        elif ground_truth:
            gt_positions = np.array([[obj['x'], obj['y']] for obj in ground_truth])
            pred_positions = np.empty((0, 2))
        elif predictions:
            gt_positions = np.empty((0, 2))
            pred_positions = np.array([[obj['x'], obj['y']] for obj in predictions])
        else:
            gt_positions = np.empty((0, 2))
            pred_positions = np.empty((0, 2))
        
        # Update OSPA
        ospa_results = self.ospa_metric.compute(gt_positions, pred_positions)
        
        # Update MOTA
        self.mota_metrics.update(ground_truth, predictions)
        
        # Update fragmentation
        self.fragmentation_metrics.update(ground_truth, predictions, frame_id)
        
        # Update maritime metrics
        self.maritime_metrics.update(
            ground_truth, predictions,
            kwargs.get('clutter_labels'),
            kwargs.get('clutter_predictions')
        )
        
        # Store frame results
        frame_result = {
            'frame_id': frame_id,
            'n_gt': len(ground_truth),
            'n_pred': len(predictions),
            **ospa_results
        }
        self.frame_results.append(frame_result)
    
    def compute_final_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Compute final comprehensive metrics.
        
        Returns:
            metrics: Dictionary with all computed metrics
        """
        # OSPA over time
        frame_ospa = [result['ospa'] for result in self.frame_results]
        avg_ospa = np.mean(frame_ospa) if frame_ospa else 0.0
        
        # MOTA metrics
        mota_results = self.mota_metrics.compute()
        
        # Fragmentation metrics
        fragmentation_results = self.fragmentation_metrics.compute()
        
        # Maritime metrics
        maritime_results = self.maritime_metrics.compute()
        
        # Combine all metrics
        final_metrics = {
            'avg_ospa': avg_ospa,
            'ospa_std': np.std(frame_ospa) if frame_ospa else 0.0,
            **mota_results,
            **fragmentation_results,
            **maritime_results,
            'total_frames': len(self.frame_results)
        }
        
        return final_metrics
    
    def generate_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            report: Formatted evaluation report
        """
        metrics = self.compute_final_metrics()
        
        report = """
=== Maritime Radar Tracking Evaluation Report ===

TRACKING ACCURACY:
- Average OSPA Distance: {avg_ospa:.2f} m
- OSPA Standard Deviation: {ospa_std:.2f} m
- MOTA (Multiple Object Tracking Accuracy): {mota:.3f}
- MOTP (Multiple Object Tracking Precision): {motp:.2f} m

DETECTION PERFORMANCE:
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- False Positive Rate: {false_positive_rate:.3f}
- ID Switches: {id_switches}

TRACK QUALITY:
- Average Track Length: {avg_track_length:.1f} frames
- Track Fragmentation Rate: {track_fragmentation_rate:.3f}
- Track Completion Rate: {track_completion_rate:.3f}
- Longest Track: {longest_track_length} frames
- Total Tracks: {total_tracks}

MARITIME-SPECIFIC METRICS:
- Target Classification Precision: {target_precision:.3f}
- Target Classification Recall: {target_recall:.3f}
- Clutter Classification Precision: {clutter_precision:.3f}
- Clutter Classification Recall: {clutter_recall:.3f}
- Average Velocity Error: {avg_velocity_error_ms:.2f} m/s
- Average Range Error: {avg_range_error_m:.2f} m
- Average Azimuth Error: {avg_azimuth_error_deg:.2f} degrees

SUMMARY:
- Total Frames Processed: {total_frames}
- Total Ground Truth Objects: {total_gt}
- Total Matched Objects: {total_matched}
- Total Missed Objects: {total_missed}
- Total False Positives: {total_false_positives}
        """.format(**metrics)
        
        return report
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot evaluation metrics over time.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.frame_results:
            print("No frame results to plot")
            return
        
        df = pd.DataFrame(self.frame_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # OSPA over time
        axes[0, 0].plot(df['frame_id'], df['ospa'])
        axes[0, 0].set_title('OSPA Distance Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('OSPA Distance (m)')
        axes[0, 0].grid(True)
        
        # Cardinality errors
        axes[0, 1].plot(df['frame_id'], df['cardinality'])
        axes[0, 1].set_title('Cardinality Error Over Time')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Cardinality Error')
        axes[0, 1].grid(True)
        
        # Localization errors
        axes[1, 0].plot(df['frame_id'], df['localization'])
        axes[1, 0].set_title('Localization Error Over Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Localization Error (m)')
        axes[1, 0].grid(True)
        
        # Detection counts
        axes[1, 1].plot(df['frame_id'], df['n_gt'], label='Ground Truth', alpha=0.7)
        axes[1, 1].plot(df['frame_id'], df['n_pred'], label='Predictions', alpha=0.7)
        axes[1, 1].set_title('Detection Counts Over Time')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Number of Objects')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()