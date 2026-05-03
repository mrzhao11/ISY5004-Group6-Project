import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Stage1Output:
    """Stage 1 output data format"""
    sequence_id: str
    behavior_label: str
    behavior_embedding: List[float]
    behavior_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "behavior_label": self.behavior_label,
            "behavior_embedding": self.behavior_embedding,
            "behavior_probabilities": self.behavior_probabilities
        }
    
    def to_csv_row(self) -> List:
        return [
            self.sequence_id,
            self.behavior_label,
            json.dumps(self.behavior_embedding),
            json.dumps(self.behavior_probabilities)
        ]


class CNNLSTMBehaviorRecognizer(nn.Module):
    """CNN + LSTM behavior recognition model"""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        embedding_dim: int = 64,
        target_length: int = 32
    ):
        super(CNNLSTMBehaviorRecognizer, self).__init__()
        
        self.target_length = target_length
        
        # CNN layers for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Embedding projection for downstream tasks
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        self.behavior_classes = ['walking', 'standing', 'looking', 'stopped']
    
    def _adaptive_pool_1d(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Adaptive pooling using interpolation for ONNX compatibility"""
        return nn.functional.interpolate(
            x, size=target_length, mode='linear', align_corners=False
        )
    
    def forward(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic sequence length support"""
        batch_size, seq_len, input_dim = trajectory.shape
        
        # CNN processing
        cnn_input = trajectory.permute(0, 2, 1)
        cnn_features = self.cnn(cnn_input)
        
        # Adaptive pooling to fixed length
        current_length = cnn_features.shape[2]
        if current_length != self.target_length:
            cnn_features = self._adaptive_pool_1d(cnn_features, self.target_length)
        
        # LSTM processing
        lstm_input = cnn_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Outputs
        logits = self.classifier(last_hidden)
        probabilities = torch.softmax(logits, dim=-1)
        embedding = self.embedding_proj(last_hidden)
        
        return logits, embedding, probabilities
    
    def predict(self, trajectory: np.ndarray, sequence_id: str) -> Stage1Output:
        """Predict behavior for a single trajectory"""
        self.eval()
        trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0)
        
        with torch.no_grad():
            logits, embedding, probabilities = self.forward(trajectory_tensor)
        
        pred_class_idx = torch.argmax(logits, dim=-1).item()
        behavior_label = self.behavior_classes[pred_class_idx]
        
        probs_dict = {
            self.behavior_classes[i]: probabilities[0][i].item()
            for i in range(len(self.behavior_classes))
        }
        
        return Stage1Output(
            sequence_id=sequence_id,
            behavior_label=behavior_label,
            behavior_embedding=embedding[0].tolist(),
            behavior_probabilities=probs_dict
        )
    
    def save_pt(self, filepath: str, metadata: Optional[Dict] = None):
        """Save PyTorch model"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.cnn[0].in_channels,
                'hidden_dim': self.lstm.hidden_size,
                'num_layers': self.lstm.num_layers,
                'num_classes': self.classifier[-1].out_features,
                'embedding_dim': self.embedding_proj.out_features,
                'behavior_classes': self.behavior_classes,
                'target_length': self.target_length
            },
            'model_class': self.__class__.__name__
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_pt(cls, filepath: str, device: str = 'cpu'):
        """Load PyTorch model"""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            embedding_dim=config['embedding_dim'],
            target_length=config.get('target_length', 32)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', None)
        print(f"✓ Model loaded from {filepath}")
        return model, metadata


class JAADDataLoader:
    """JAAD dataset loader - adapted for your data format"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Root data directory (e.g., 'data/processed/jaad/' or 'data/processed/mock_jaad/')
        """
        self.data_path = Path(data_path)
        
        # Load three core files
        self.metadata = pd.read_csv(self.data_path / 'cleaned_metadata.csv')
        self.sequences_manifest = pd.read_csv(self.data_path / 'cleaned_sequences_manifest.csv')
        self.trajectory_features = pd.read_csv(self.data_path / 'trajectory_features.csv')
        
        print(f"✓ Loaded JAAD dataset from {data_path}")
        print(f"  - Metadata: {len(self.metadata)} rows")
        print(f"  - Sequences manifest: {len(self.sequences_manifest)} rows")
        print(f"  - Trajectory features: {len(self.trajectory_features)} rows")
        
        # Display column names for debugging
        print(f"\n  Columns in metadata: {list(self.metadata.columns)}")
        print(f"  Columns in sequences_manifest: {list(self.sequences_manifest.columns)}")
        print(f"  Columns in trajectory_features: {list(self.trajectory_features.columns)}")
        
        # Preprocess metadata and build trajectory cache
        self._build_trajectory_cache()
    
    def _build_trajectory_cache(self):
        """Build trajectory cache from metadata"""
        print("\n  Building trajectory cache from metadata...")
        
        # Check if metadata contains position information
        if 'xtl' in self.metadata.columns and 'ytl' in self.metadata.columns:
            # Use bounding box top-left corner as trajectory point
            # Alternatively use center point: (xtl + width/2, ytl + height/2)
            self.metadata['center_x'] = self.metadata['xtl'] + self.metadata['width'] / 2
            self.metadata['center_y'] = self.metadata['ytl'] + self.metadata['height'] / 2
            
            # Group by sequence_id
            if 'sequence_id' not in self.metadata.columns:
                # If sequence_id doesn't exist, use video_id + pedestrian_id combination
                print("  Warning: 'sequence_id' not found in metadata, using video_id+pedestrian_id")
                self.metadata['sequence_id'] = self.metadata['video_id'] + '__' + self.metadata['pedestrian_id'].astype(str)
            
            # Sort by sequence_id and frame_id
            self.metadata_sorted = self.metadata.sort_values(['sequence_id', 'frame_id'])
            
            # Group and store trajectories
            self.trajectory_cache = {}
            for seq_id, group in self.metadata_sorted.groupby('sequence_id'):
                # Extract trajectory coordinates (using center point)
                traj = group[['center_x', 'center_y']].values.astype(np.float32)
                if len(traj) >= 5:  # Minimum 5 frames
                    self.trajectory_cache[str(seq_id)] = traj
            
            print(f"  ✓ Built cache with {len(self.trajectory_cache)} trajectories")
        else:
            print("  Warning: No position data (xtl/ytl) found in metadata")
            self.trajectory_cache = {}
    
    def get_available_sequences(self) -> List[str]:
        """Get all available sequence IDs"""
        # Priority: use sequences from cache
        if hasattr(self, 'trajectory_cache') and self.trajectory_cache:
            return list(self.trajectory_cache.keys())
        
        # Otherwise get from sequences_manifest
        if 'sequence_id' in self.sequences_manifest.columns:
            return self.sequences_manifest['sequence_id'].astype(str).unique().tolist()
        elif 'seq_id' in self.sequences_manifest.columns:
            return self.sequences_manifest['seq_id'].astype(str).unique().tolist()
        else:
            possible_cols = ['id', 'seq_id', 'sequence_id', 'track_id']
            for col in possible_cols:
                if col in self.sequences_manifest.columns:
                    return self.sequences_manifest[col].astype(str).unique().tolist()
            return []
    
    def get_trajectory_by_sequence_id(self, sequence_id: Union[str, int]) -> Optional[np.ndarray]:
        """
        Get trajectory data by sequence_id
        
        Returns:
            trajectory: numpy array of shape (seq_len, 2) containing x, y coordinates
        """
        seq_id_str = str(sequence_id)
        
        # Priority: get from cache
        if hasattr(self, 'trajectory_cache') and seq_id_str in self.trajectory_cache:
            return self.trajectory_cache[seq_id_str]
        
        # If not in cache, try to generate synthetic trajectory from trajectory_features (for demo)
        # Note: trajectory_features only has statistical features, not coordinates
        # Generate a synthetic trajectory for demonstration
        if seq_id_str in self.trajectory_features['sequence_id'].astype(str).values:
            # Get features for this sequence
            row = self.trajectory_features[self.trajectory_features['sequence_id'].astype(str) == seq_id_str].iloc[0]
            
            # Generate synthetic trajectory (in production, use actual coordinate data)
            traj_len = min(int(row.get('trajectory_length', 30)), 50)
            traj_len = max(traj_len, 10)
            
            # Generate synthetic trajectory based on motion direction
            direction = row.get('motion_direction', 'right')
            speed = row.get('speed_mean', 5)
            
            # Generate random but continuous trajectory
            t = np.linspace(0, traj_len, traj_len)
            if 'right' in str(direction).lower():
                x = t * speed + np.random.randn(traj_len) * 2
            else:
                x = -t * speed + np.random.randn(traj_len) * 2
            
            y = np.sin(t * 0.3) * 20 + np.random.randn(traj_len) * 3 + 100
            
            traj = np.column_stack([x, y]).astype(np.float32)
            return traj
        
        print(f"⚠ No trajectory data found for sequence_id: {sequence_id}")
        return None
    
    def get_behavior_label(self, sequence_id: Union[str, int]) -> Optional[str]:
        """Get behavior label for a sequence"""
        seq_id_str = str(sequence_id)
        
        # Get from sequences_manifest
        if 'sequence_id' in self.sequences_manifest.columns:
            mask = self.sequences_manifest['sequence_id'].astype(str) == seq_id_str
        elif 'seq_id' in self.sequences_manifest.columns:
            mask = self.sequences_manifest['seq_id'].astype(str) == seq_id_str
        else:
            mask = self.sequences_manifest.iloc[:, 0].astype(str) == seq_id_str
        
        if mask.any():
            # Try to get behavior label
            label_cols = ['sequence_behavior_label', 'behavior_label', 'action', 'label']
            for col in label_cols:
                if col in self.sequences_manifest.columns:
                    label = self.sequences_manifest.loc[mask, col].values[0]
                    if pd.notna(label):
                        return self._map_label_to_classes(label)
        
        # Get from metadata
        if 'sequence_id' in self.metadata.columns:
            mask = self.metadata['sequence_id'].astype(str) == seq_id_str
        elif 'pedestrian_id' in self.metadata.columns and 'video_id' in self.metadata.columns:
            # Try to parse video_id and pedestrian_id from sequence_id
            parts = seq_id_str.split('__')
            if len(parts) >= 2:
                video_id = parts[0]
                ped_id = parts[1]
                mask = (self.metadata['video_id'] == video_id) & (self.metadata['pedestrian_id'].astype(str) == ped_id)
            else:
                mask = pd.Series([False] * len(self.metadata))
        else:
            mask = pd.Series([False] * len(self.metadata))
        
        if mask.any():
            label_cols = ['behavior_label', 'crossing_label', 'action', 'label']
            for col in label_cols:
                if col in self.metadata.columns:
                    # Take the most common label
                    labels = self.metadata.loc[mask, col].dropna()
                    if len(labels) > 0:
                        label = labels.mode()[0] if len(labels) > 0 else labels.iloc[0]
                        return self._map_label_to_classes(label)
        
        return None
    
    def _map_label_to_classes(self, label: str) -> str:
        """Map original labels to four behavior classes"""
        label_lower = str(label).lower()
        
        walking_keywords = ['walk', 'walking', 'moving', 'cross', 'crossing']
        standing_keywords = ['stand', 'standing', 'wait', 'waiting']
        looking_keywords = ['look', 'looking', 'watch', 'watching', 'gaze']
        stopped_keywords = ['stop', 'stopped', 'idle', 'static', 'still']
        
        for kw in walking_keywords:
            if kw in label_lower:
                return 'walking'
        for kw in standing_keywords:
            if kw in label_lower:
                return 'standing'
        for kw in looking_keywords:
            if kw in label_lower:
                return 'looking'
        for kw in stopped_keywords:
            if kw in label_lower:
                return 'stopped'
        
        # Default return
        return 'walking'
    
    def load_all_trajectories(self, min_length: int = 10, max_length: int = None) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Load all trajectories
        
        Returns:
            trajectories: List of trajectory arrays (each shape: seq_len, 2)
            sequence_ids: List of sequence IDs
            behavior_labels: List of behavior labels (for evaluation)
        """
        sequence_ids = self.get_available_sequences()
        
        trajectories = []
        valid_ids = []
        valid_labels = []
        
        for seq_id in sequence_ids:
            traj = self.get_trajectory_by_sequence_id(seq_id)
            
            if traj is None or len(traj) < min_length:
                continue
            
            if max_length and len(traj) > max_length:
                # Optionally truncate
                traj = traj[:max_length]
            
            label = self.get_behavior_label(seq_id)
            
            trajectories.append(traj)
            valid_ids.append(str(seq_id))
            valid_labels.append(label if label else 'unknown')
        
        print(f"✓ Loaded {len(trajectories)} trajectories (min_length={min_length})")
        
        # Count label distribution
        label_counts = {}
        for label in valid_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Label distribution: {label_counts}")
        
        return trajectories, valid_ids, valid_labels
    
    def get_trajectory_features_df(self) -> pd.DataFrame:
        """Return trajectory features DataFrame for exploration"""
        return self.trajectory_features


class Stage1Processor:
    """Stage 1 Data Processor - Main interface for behavior recognition"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.model, _ = CNNLSTMBehaviorRecognizer.load_pt(model_path, device)
            self.model.to(device)
            self.model.eval()
            print(f"Using PyTorch model: {model_path}")
        else:
            print("No model loaded. Use train_model() first or provide model_path.")
    
    def process_trajectory(
        self,
        trajectory: np.ndarray,
        sequence_id: str,
        output_format: str = "dict"
    ) -> Union[Dict, str, List]:
        """Process a single trajectory"""
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(f"Trajectory must be of shape (seq_len, 2), got {trajectory.shape}")
        
        if self.model is None:
            raise ValueError("No model loaded. Please provide model_path when initializing processor.")
        
        output = self.model.predict(trajectory, sequence_id)
        
        if output_format == "dict":
            return output.to_dict()
        elif output_format == "json":
            return json.dumps(output.to_dict(), indent=2)
        elif output_format == "csv":
            return output.to_csv_row()
        else:
            return output.to_dict()
    
    def process_to_dataframe(
        self,
        trajectories: List[np.ndarray],
        sequence_ids: List[str],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Process trajectories and return as DataFrame in required format"""
        results = []
        for traj, seq_id in zip(trajectories, sequence_ids):
            result = self.process_trajectory(traj, seq_id, "dict")
            results.append(result)
        
        # Create DataFrame in required format
        rows = []
        for result in results:
            row = {'sequence_id': result['sequence_id']}
            
            # Expand embedding (64 dimensions)
            for j, val in enumerate(result['behavior_embedding']):
                row[f'beh_emb_{j}'] = val
            
            # Expand probabilities
            probs = result['behavior_probabilities']
            row['p_walk'] = probs.get('walking', 0.0)
            row['p_stand'] = probs.get('standing', 0.0)
            row['p_look'] = probs.get('looking', 0.0)
            row['p_stop'] = probs.get('stopped', 0.0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Results saved to {output_file}")
        
        return df
    
    def process_dataset(
        self,
        data_loader: JAADDataLoader,
        min_length: int = 10,
        output_file: Optional[str] = None,
        save_embeddings: bool = True
    ) -> pd.DataFrame:
        """
        Process the entire dataset
        
        Args:
            data_loader: JAADDataLoader instance
            min_length: Minimum trajectory length
            output_file: Output CSV file path
            save_embeddings: Whether to save embedding vectors to .npy file
        """
        print("\n" + "=" * 60)
        print("Processing JAAD Dataset with Stage 1 Model")
        print("=" * 60)
        
        # Load all trajectories
        trajectories, sequence_ids, true_labels = data_loader.load_all_trajectories(min_length=min_length)
        
        if len(trajectories) == 0:
            print("No valid trajectories found!")
            return pd.DataFrame()
        
        # Batch processing
        print(f"\nProcessing {len(trajectories)} trajectories...")
        results = []
        for i, (traj, seq_id) in enumerate(zip(trajectories, sequence_ids)):
            result = self.process_trajectory(traj, seq_id, "dict")
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(trajectories)}")
        
        # Create DataFrame in required format
        print("\nCreating DataFrame in required format...")
        rows = []
        for i, result in enumerate(results):
            row = {}
            
            # 1. sequence_id
            row['sequence_id'] = result['sequence_id']
            
            # 2. behavior embedding (expand to 64 separate columns)
            embedding = result['behavior_embedding']
            for j, val in enumerate(embedding):
                row[f'beh_emb_{j}'] = val
            
            # 3. behavior probabilities (expand to 4 separate columns)
            probs = result['behavior_probabilities']
            row['p_walk'] = probs.get('walking', 0.0)
            row['p_stand'] = probs.get('standing', 0.0)
            row['p_look'] = probs.get('looking', 0.0)
            row['p_stop'] = probs.get('stopped', 0.0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Optional: Add extra information for debugging (doesn't affect Stage 2 usage)
        df['behavior_label'] = [r['behavior_label'] for r in results]
        df['true_label'] = true_labels
        df['trajectory_length'] = [len(t) for t in trajectories]
        
        print(f"\n✓ Processing complete!")
        print(f"  Total trajectories: {len(df)}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:5])} ... beh_emb_63, p_walk, p_stand, p_look, p_stop")
        
        # Calculate accuracy for validation
        df['correct'] = df['behavior_label'] == df['true_label']
        accuracy = df['correct'].mean()
        print(f"  Accuracy (vs true labels): {accuracy:.2%}")
        
        # Save results to CSV (only save required format columns)
        if output_file:
            output_columns = ['sequence_id'] + [f'beh_emb_{j}' for j in range(64)] + ['p_walk', 'p_stand', 'p_look', 'p_stop']
            df[output_columns].to_csv(output_file, index=False)
            print(f"  Results saved to: {output_file}")
        
        # Save embeddings for downstream tasks
        if save_embeddings:
            embeddings = np.vstack([r['behavior_embedding'] for r in results])
            embedding_file = output_file.replace('.csv', '_embeddings.npy') if output_file else "embeddings.npy"
            np.save(embedding_file, embeddings)
            print(f"  Embeddings saved to: {embedding_file}")
        
        # Print confusion matrix
        if accuracy > 0:
            self._print_confusion_matrix(df)
        
        return df
    
    def _print_confusion_matrix(self, df: pd.DataFrame):
        """Print confusion matrix"""
        labels = ['walking', 'standing', 'looking', 'stopped']
        conf_matrix = {}
        
        for true_label in labels:
            conf_matrix[true_label] = {}
            for pred_label in labels:
                conf_matrix[true_label][pred_label] = 0
        
        for _, row in df.iterrows():
            true = row['true_label']
            pred = row['behavior_label']
            if true in conf_matrix and pred in conf_matrix[true]:
                conf_matrix[true][pred] += 1
        
        print("\n  Confusion Matrix:")
        header = " " * 12 + "".join([f"{l:10}" for l in labels])
        print(header)
        for true_label in labels:
            row_str = f"{true_label:12}"
            for pred_label in labels:
                row_str += f"{conf_matrix[true_label].get(pred_label, 0):10}"
            print(row_str)


class SafeJSONEncoder(json.JSONEncoder):
    """Safe JSON encoder for handling non-serializable objects"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)


class Stage1OutputInterface:
    """Standardized interface for Stage 1 output to be consumed by Stage 2"""
    
    @staticmethod
    def to_downstream_format(stage1_output: Dict) -> Dict:
        """Convert Stage 1 output to format expected by Stage 2"""
        return {
            "type": "behavior_features",
            "source": "stage1",
            "data": {
                "sequence_id": stage1_output["sequence_id"],
                "behavior_label": stage1_output["behavior_label"],
                "behavior_embedding": stage1_output["behavior_embedding"],
                "behavior_probabilities": stage1_output["behavior_probabilities"],
                "confidence": max(stage1_output["behavior_probabilities"].values())
            }
        }
    
    @staticmethod
    def to_message_format(stage1_output: Dict) -> str:
        """Convert to human-readable message format"""
        probs = stage1_output["behavior_probabilities"]
        top_behavior = stage1_output["behavior_label"]
        confidence = probs[top_behavior]
        return f"Pedestrian {stage1_output['sequence_id']} is {top_behavior} (confidence: {confidence:.2%})"


def train_demo_model(save_path: str = "models/jaad_model.pt"):
    """Train a demo model (using random weights)"""
    os.makedirs("models", exist_ok=True)
    model = CNNLSTMBehaviorRecognizer()
    model.save_pt(save_path, metadata={"demo": True, "dataset": "JAAD"})
    print(f"Demo model saved to {save_path}")
    return model


def main():
    """Main function - use your JAAD dataset"""
    print("=" * 60)
    print("STAGE 1 BEHAVIOR RECOGNITION - JAAD DATASET")
    print("=" * 60)
    
    # Get the project root directory (assuming script is in models/stage1/)
    script_dir = Path(__file__).parent  # models/stage1/
    project_root = script_dir.parent.parent  # Project root
    
    print(f"\n📁 Script directory: {script_dir}")
    print(f"📁 Project root: {project_root}")
    
    # 1. Select dataset path (using absolute paths)
    data_paths = [
        str(project_root / "data/processed/jaad"),
        str(project_root / "data/processed/mock_jaad")
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"\n✓ Found dataset at: {data_path}")
            break
    
    if data_path is None:
        print("\n❌ Dataset not found! Please check your data path.")
        print(f"Project root: {project_root}")
        print("Expected paths:")
        for path in data_paths:
            print(f"  - {path}")
        
        # Try to list data directory contents for debugging
        data_dir = project_root / "data" / "processed"
        if data_dir.exists():
            print(f"\n📂 Contents of {data_dir}:")
            for item in data_dir.iterdir():
                print(f"    - {item.name}")
        else:
            print(f"\n❌ {data_dir} does not exist!")
            print("   Please make sure your data files are in the correct location.")
        return
    
    # 2. Load data
    data_loader = JAADDataLoader(data_path)
    
    # 3. Check data format (optional: show examples)
    print("\n" + "-" * 40)
    print("Data Exploration:")
    print("-" * 40)
    
    # Show first few rows of trajectory_features
    print("\nFirst 5 rows of trajectory_features.csv:")
    print(data_loader.trajectory_features.head())
    
    # Show available sequences
    seq_ids = data_loader.get_available_sequences()
    print(f"\nAvailable sequences: {len(seq_ids)}")
    if len(seq_ids) > 0:
        print(f"First 5 sequence IDs: {seq_ids[:5]}")
    
    # Show an example trajectory
    if len(seq_ids) > 0:
        sample_traj = data_loader.get_trajectory_by_sequence_id(seq_ids[0])
        if sample_traj is not None:
            print(f"\nExample trajectory (first sequence):")
            print(f"  Shape: {sample_traj.shape}")
            print(f"  First 5 positions:\n{sample_traj[:5]}")
            label = data_loader.get_behavior_label(seq_ids[0])
            print(f"  Behavior label: {label}")
    
    # 4. Train/Load model
    print("\n" + "-" * 40)
    print("Model Setup:")
    print("-" * 40)
    
    # Save model to models folder in project root
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / "jaad_model.pt")
    
    if not os.path.exists(model_path):
        print("Training demo model...")
        train_demo_model(model_path)
    else:
        print(f"Model already exists at: {model_path}")
    
    # 5. Initialize processor
    processor = Stage1Processor(model_path=model_path, device='cpu')
    
    # 6. Process the entire dataset
    output_file = str(project_root / "stage1_jaad_results.csv")
    df = processor.process_dataset(
        data_loader=data_loader,
        min_length=10,  # Minimum 10 frames
        output_file=output_file,
        save_embeddings=True
    )
    
    # 7. Display result summary
    if len(df) > 0:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total processed: {len(df)} trajectories")
        
        # Use behavior_label column if it exists
        if 'behavior_label' in df.columns:
            print(f"\nPredicted behavior distribution:")
            for label, count in df['behavior_label'].value_counts().items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Calculate average confidence (from probability columns)
        prob_cols = ['p_walk', 'p_stand', 'p_look', 'p_stop']
        existing_prob_cols = [col for col in prob_cols if col in df.columns]
        if existing_prob_cols:
            max_probs = df[existing_prob_cols].max(axis=1)
            print(f"\nAverage confidence: {max_probs.mean():.3f}")
        
        # Show first few results
        print("\nFirst 5 results:")
        display_cols = ['sequence_id']
        display_cols.extend([col for col in prob_cols if col in df.columns])
        if 'behavior_label' in df.columns:
            display_cols.append('behavior_label')
        if 'true_label' in df.columns:
            display_cols.append('true_label')
        
        print(df[display_cols].head())
        
        # Show output file locations
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Embeddings saved to: {output_file.replace('.csv', '_embeddings.npy')}")
    
    print("\n" + "=" * 60)
    print("✓ Stage 1 Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
