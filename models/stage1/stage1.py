import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import os
import onnx
import onnxruntime


@dataclass
class Stage1Output:
    """Stage 1 output data format"""
    sequence_id: str          # Pedestrian trajectory ID in the video
    behavior_label: str       # Behavior class (walking/standing/looking etc.)
    behavior_embedding: List[float]  # Behavior feature vector
    behavior_probabilities: Dict[str, float]  # Probability distribution over behavior classes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "sequence_id": self.sequence_id,
            "behavior_label": self.behavior_label,
            "behavior_embedding": self.behavior_embedding,
            "behavior_probabilities": self.behavior_probabilities
        }
    
    def to_csv_row(self) -> List:
        """Convert to CSV row format"""
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
        input_dim: int = 2,          # Input feature dimension (x, y trajectory)
        hidden_dim: int = 128,       # LSTM hidden layer dimension
        num_layers: int = 2,         # Number of LSTM layers
        num_classes: int = 4,        # Number of behavior classes
        embedding_dim: int = 64      # Output embedding dimension
    ):
        super(CNNLSTMBehaviorRecognizer, self).__init__()
        
        # CNN layers: Extract local trajectory features
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # LSTM layers: Temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Behavior classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Embedding projection layer
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Behavior class mapping
        self.behavior_classes = ['walking', 'standing', 'looking', 'crossing']
        
    def forward(
        self, 
        trajectory: torch.Tensor  # shape: (batch, seq_len, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: Trajectory sequence [batch, seq_len, 2] (x, y coordinates)
        
        Returns:
            logits: Classification logits [batch, num_classes]
            embedding: Behavior embedding [batch, embedding_dim]
            probabilities: Probability distribution [batch, num_classes]
        """
        batch_size, seq_len, input_dim = trajectory.shape
        
        # CNN processing: Need dimensions (batch, input_dim, seq_len)
        cnn_input = trajectory.permute(0, 2, 1)  # [batch, 2, seq_len]
        cnn_features = self.cnn(cnn_input)        # [batch, 64, 32]
        
        # LSTM processing: Need dimensions (batch, seq_len, feature_dim)
        lstm_input = cnn_features.permute(0, 2, 1)  # [batch, 32, 64]
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Classification output
        logits = self.classifier(last_hidden)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Embedding output
        embedding = self.embedding_proj(last_hidden)
        
        return logits, embedding, probabilities
    
    def predict(
        self, 
        trajectory: np.ndarray,      # (seq_len, 2)
        sequence_id: str
    ) -> Stage1Output:
        """
        Predict behavior for a single trajectory
        
        Args:
            trajectory: Trajectory sequence (seq_len, 2)
            sequence_id: Sequence ID
            
        Returns:
            Stage1Output: Formatted output
        """
        self.eval()
        
        # Convert to tensor and add batch dimension
        trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0)  # (1, seq_len, 2)
        
        with torch.no_grad():
            logits, embedding, probabilities = self.forward(trajectory_tensor)
            
        # Get predicted class
        pred_class_idx = torch.argmax(logits, dim=-1).item()
        behavior_label = self.behavior_classes[pred_class_idx]
        
        # Convert to probability dictionary
        probs_dict = {
            self.behavior_classes[i]: probabilities[0][i].item()
            for i in range(len(self.behavior_classes))
        }
        
        # Return formatted output
        return Stage1Output(
            sequence_id=sequence_id,
            behavior_label=behavior_label,
            behavior_embedding=embedding[0].tolist(),
            behavior_probabilities=probs_dict
        )
    
    def predict_batch(
        self,
        trajectories: List[np.ndarray],
        sequence_ids: List[str]
    ) -> List[Stage1Output]:
        """Batch prediction"""
        results = []
        for traj, seq_id in zip(trajectories, sequence_ids):
            results.append(self.predict(traj, seq_id))
        return results
    
    def save_pt(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Save model in .pt format
        
        Args:
            filepath: Save path (recommended to end with .pt or .pth)
            metadata: Additional metadata information
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Prepare content to save
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.cnn[0].in_channels,
                'hidden_dim': self.lstm.hidden_size,
                'num_layers': self.lstm.num_layers,
                'num_classes': self.classifier[-1].out_features,
                'embedding_dim': self.embedding_proj.out_features,
                'behavior_classes': self.behavior_classes
            },
            'model_class': self.__class__.__name__
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, filepath)
        print(f"✓ Model saved to {filepath}")
        
        # Also save a config file
        config_path = filepath.replace('.pt', '_config.json').replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(save_dict['model_config'], f, indent=2)
        print(f"✓ Config saved to {config_path}")
    
    def save_onnx(
        self,
        filepath: str,
        seq_len: int = 30,
        input_dim: int = 2,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None
    ):
        """
        Save model in ONNX format
        
        Args:
            filepath: Save path (recommended to end with .onnx)
            seq_len: Input sequence length
            input_dim: Input feature dimension
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration, e.g., {'input': {0: 'batch_size', 1: 'seq_len'}}
        """
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, seq_len, input_dim)
        
        # Default dynamic axes configuration
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'seq_len'},
                'logits': {0: 'batch_size'},
                'embedding': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'embedding', 'probabilities'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Validate ONNX model
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model saved to {filepath}")
        print(f"✓ ONNX model validation passed")
        
        # Save model configuration
        config_path = filepath.replace('.onnx', '_config.json')
        config = {
            'input_seq_len': seq_len,
            'input_dim': input_dim,
            'output_names': ['logits', 'embedding', 'probabilities'],
            'opset_version': opset_version,
            'behavior_classes': self.behavior_classes
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {config_path}")
    
    @classmethod
    def load_pt(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from .pt file
        
        Args:
            filepath: Model file path
            device: Device ('cpu' or 'cuda')
            
        Returns:
            model: Loaded model instance
            metadata: Metadata (if any)
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model from config
        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            embedding_dim=config['embedding_dim']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        metadata = checkpoint.get('metadata', None)
        print(f"✓ Model loaded from {filepath}")
        
        return model, metadata
    
    def test_onnx_inference(self, sample_input: np.ndarray):
        """
        Test ONNX model inference
        
        Args:
            sample_input: Example input (batch_size, seq_len, 2)
        """
        # Save temporary ONNX file
        temp_path = "temp_test.onnx"
        self.save_onnx(temp_path, seq_len=sample_input.shape[1])
        
        # Inference using ONNX Runtime
        ort_session = onnxruntime.InferenceSession(temp_path)
        
        # Prepare input
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.astype(np.float32)}
        
        # Inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # PyTorch inference
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sample_input)
            logits, embedding, probs = self.forward(input_tensor)
        
        # Compare results
        print("\n=== ONNX vs PyTorch Inference Comparison ===")
        print(f"PyTorch logits shape: {logits.shape}")
        print(f"ONNX logits shape: {ort_outputs[0].shape}")
        print(f"Max difference: {np.max(np.abs(logits.numpy() - ort_outputs[0])):.6f}")
        
        # Clean up temporary files
        os.remove(temp_path)
        config_path = temp_path.replace('.onnx', '_config.json')
        if os.path.exists(config_path):
            os.remove(config_path)
        
        return ort_outputs


class Stage1Processor:
    """Stage 1 Data Processor - Responsible for extracting pedestrian trajectories from video and outputting behavior features"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Args:
            model_path: Pretrained model weights path
            device: Device ('cpu' or 'cuda')
        """
        self.device = device
        self.model = CNNLSTMBehaviorRecognizer()
        
        if model_path and os.path.exists(model_path):
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                self.model, _ = CNNLSTMBehaviorRecognizer.load_pt(model_path, device)
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
    def process_trajectory(
        self,
        trajectory: np.ndarray,      # (seq_len, 2) trajectory coordinates
        sequence_id: str,
        output_format: str = "dict"   # "dict", "csv", "json"
    ) -> Dict:
        """
        Process a single trajectory and output formatted results
        
        Args:
            trajectory: Pedestrian trajectory sequence
            sequence_id: Trajectory ID
            output_format: Output format (dict/csv/json)
            
        Returns:
            Results in the specified output format
        """
        output = self.model.predict(trajectory, sequence_id)
        
        if output_format == "dict":
            return output.to_dict()
        elif output_format == "json":
            return json.dumps(output.to_dict(), indent=2)
        elif output_format == "csv":
            return output.to_csv_row()
        else:
            return output.to_dict()
    
    def process_video_batch(
        self,
        trajectories: List[np.ndarray],
        sequence_ids: List[str],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Batch process all trajectories extracted from the video
        
        Args:
            trajectories: List of trajectories
            sequence_ids: Corresponding ID list
            output_file: Output CSV file path (optional)
            
        Returns:
            DataFrame: Table containing all outputs
        """
        results = self.model.predict_batch(trajectories, sequence_ids)
        
        # Convert to DataFrame
        rows = []
        for r in results:
            rows.append({
                'sequence_id': r.sequence_id,
                'behavior_label': r.behavior_label,
                'behavior_embedding': json.dumps(r.behavior_embedding),
                'behavior_probabilities': json.dumps(r.behavior_probabilities)
            })
        
        df = pd.DataFrame(rows)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
        
        return df
    
    def get_output_spec(self) -> Dict:
        """
        Return Stage 1 output format specification for downstream modules
        
        Returns:
            Output format description dictionary
        """
        return {
            "description": "Stage 1 (Behavior Recognition) Output Format",
            "format": {
                "sequence_id": {
                    "type": "str",
                    "description": "Unique identifier for each pedestrian trajectory in the video"
                },
                "behavior_label": {
                    "type": "str",
                    "description": "Predicted behavior class (e.g., walking, standing, looking, crossing)"
                },
                "behavior_embedding": {
                    "type": "List[float]",
                    "dimension": 64,
                    "description": "Dense feature vector representing behavior characteristics"
                },
                "behavior_probabilities": {
                    "type": "Dict[str, float]",
                    "description": "Probability distribution over all behavior classes"
                }
            },
            "example_output": {
                "sequence_id": "video_001_ped_001",
                "behavior_label": "crossing",
                "behavior_embedding": [0.12, -0.34, 0.56, ...],  # 64-dim vector
                "behavior_probabilities": {
                    "walking": 0.05,
                    "standing": 0.02,
                    "looking": 0.08,
                    "crossing": 0.85
                }
            },
            "csv_export_format": [
                "sequence_id",
                "behavior_label", 
                "behavior_embedding (JSON string)",
                "behavior_probabilities (JSON string)"
            ]
        }
    
    def save_models(self, base_path: str, seq_len: int = 30):
        """
        Save models in multiple formats
        
        Args:
            base_path: Base path (without extension)
            seq_len: Input sequence length for ONNX model
        """
        # Save as .pt format
        pt_path = f"{base_path}.pt"
        metadata = {
            "model_type": "CNNLSTMBehaviorRecognizer",
            "input_seq_len": seq_len,
            "input_dim": 2,
            "save_date": str(pd.Timestamp.now()),
            "device": self.device
        }
        self.model.save_pt(pt_path, metadata)
        
        # Save as .onnx format
        onnx_path = f"{base_path}.onnx"
        self.model.save_onnx(onnx_path, seq_len=seq_len)
        
        print(f"\n✓ All models saved successfully!")
        print(f"  - PyTorch model: {pt_path}")
        print(f"  - ONNX model: {onnx_path}")


# Training simulation function (for demonstration)
def train_demo_model(num_samples: int = 1000, seq_len: int = 30):
    """
    Train a demonstration model (using random data)
    
    Args:
        num_samples: Number of training samples
        seq_len: Sequence length
    """
    print("Starting demo training...")
    
    # Create model
    model = CNNLSTMBehaviorRecognizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Generate synthetic training data
    print(f"Generating {num_samples} training samples...")
    trajectories = []
    labels = []
    
    for i in range(num_samples):
        # Generate different trajectory patterns based on labels
        label = np.random.randint(0, 4)
        labels.append(label)
        
        if label == 0:  # walking - linear motion
            traj = np.cumsum(np.random.randn(seq_len, 2) * 2 + [0.5, 0], axis=0)
        elif label == 1:  # standing - small range movement
            traj = np.random.randn(seq_len, 2) * 0.5
            traj = np.cumsum(traj, axis=0)
        elif label == 2:  # looking - back and forth movement
            traj = np.cumsum(np.random.randn(seq_len, 2) * 3, axis=0)
            traj[:, 0] = np.sin(np.linspace(0, 4*np.pi, seq_len)) * 10
        else:  # crossing - fast crossing motion
            traj = np.cumsum(np.random.randn(seq_len, 2) * 4 + [1, 0.5], axis=0)
        
        trajectories.append(traj)
    
    # Training loop
    print("Training for 10 epochs...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for traj, label in zip(trajectories, labels):
            traj_tensor = torch.FloatTensor(traj).unsqueeze(0)
            label_tensor = torch.LongTensor([label])
            
            optimizer.zero_grad()
            logits, _, _ = model(traj_tensor)
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss/len(trajectories):.4f}")
    
    print("Training completed!")
    return model


# Usage example
if __name__ == "__main__":
    # Simulate trajectory data
    seq_len = 30  # 30 frames of trajectory
    sample_trajectory = np.cumsum(np.random.randn(seq_len, 2) * 5, axis=0)
    
    # Train a demonstration model (in practice, you would train with real data)
    print("=" * 60)
    print("TRAINING DEMO MODEL")
    print("=" * 60)
    trained_model = train_demo_model(num_samples=500, seq_len=seq_len)
    
    # Initialize Stage 1 processor
    processor = Stage1Processor()
    processor.model = trained_model
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    processor.save_models("models/behavior_recognition_model", seq_len=seq_len)
    
    # View output format specification
    print("\n" + "=" * 60)
    print("OUTPUT FORMAT SPECIFICATION")
    print("=" * 60)
    spec = processor.get_output_spec()
    print(json.dumps(spec, indent=2, default=str))
    
    # Single trajectory processing example
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUT (Single Trajectory)")
    print("=" * 60)
    result = processor.process_trajectory(
        trajectory=sample_trajectory,
        sequence_id="video_001_pedestrian_001",
        output_format="dict"
    )
    print(json.dumps(result, indent=2))
    
    # Batch processing example
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUT (Batch Processing)")
    print("=" * 60)
    trajectories = [
        np.cumsum(np.random.randn(seq_len, 2) * 5, axis=0),
        np.cumsum(np.random.randn(seq_len, 2) * 4, axis=0),
        np.cumsum(np.random.randn(seq_len, 2) * 6, axis=0)
    ]
    ids = ["vid1_ped1", "vid1_ped2", "vid2_ped1"]
    
    df = processor.process_video_batch(trajectories, ids)
    print(df.to_string())
    
    # Test loading saved model
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    loaded_model, metadata = CNNLSTMBehaviorRecognizer.load_pt("models/behavior_recognition_model.pt")
    print(f"Loaded model metadata: {metadata}")
    
    # Test ONNX inference (optional, requires onnxruntime)
    try:
        print("\n" + "=" * 60)
        print("TESTING ONNX INFERENCE")
        print("=" * 60)
        test_input = np.random.randn(1, seq_len, 2).astype(np.float32)
        onnx_outputs = loaded_model.test_onnx_inference(test_input)
        print("✓ ONNX inference test passed!")
    except Exception as e:
        print(f"ONNX test skipped: {e}")
