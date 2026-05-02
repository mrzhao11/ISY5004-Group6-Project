import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import os
import warnings

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
    """CNN + LSTM behavior recognition model with dynamic ONNX export support"""
    
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
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # LSTM layers
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
        
        # Embedding projection
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        
        self.behavior_classes = ['walking', 'standing', 'looking', 'crossing']
    
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
        
        # Adaptive pooling to fixed length (use constant for ONNX compatibility)
        # 使用 torch.where 避免动态条件
        current_length = cnn_features.shape[2]
        
        # 对于 ONNX 导出，使用固定操作
        if self.training or not torch.jit.is_tracing():
            if current_length != self.target_length:
                cnn_features = self._adaptive_pool_1d(cnn_features, self.target_length)
        else:
            # 导出时使用固定池化
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
        """Predict for single trajectory"""
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
    
    def save_onnx_dynamic(
        self,
        filepath: str,
        input_seq_len: int = 30,
        input_dim: int = 2,
        opset_version: int = 14,
        use_dynamic_batch: bool = True,
        use_dynamic_seq_len: bool = True
    ):
        """
        Export ONNX with dynamic axes support
        """
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_seq_len, input_dim)
        
        # Configure dynamic axes
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
        
        if use_dynamic_seq_len:
            dynamic_axes['input'][1] = 'sequence_length'
        
        print(f"Exporting ONNX with dynamic axes: {dynamic_axes}")
        print(f"  - Dynamic batch: {use_dynamic_batch}")
        print(f"  - Dynamic sequence length: {use_dynamic_seq_len}")
        
        # 使用 torch.jit.trace 来避免 tracing 问题
        try:
            # 导出 ONNX
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
            try:
                import onnx
                onnx_model = onnx.load(filepath)
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model saved to {filepath}")
                print(f"✓ ONNX model validation passed")
            except ImportError:
                print(f"✓ ONNX model saved to {filepath} (validation skipped)")
            except Exception as e:
                print(f"⚠ ONNX model saved but validation warning: {e}")
            
            # Save config
            config = {
                'input_seq_len': input_seq_len,
                'input_dim': input_dim,
                'output_names': ['logits', 'embedding', 'probabilities'],
                'opset_version': opset_version,
                'behavior_classes': self.behavior_classes,
                'target_length': self.target_length,
                'dynamic_axes': {
                    'batch_size': 'variable',
                    'sequence_length': 'variable' if use_dynamic_seq_len else 'fixed'
                }
            }
            
            config_path = filepath.replace('.onnx', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Config saved to {config_path}")
            
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            raise
    
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
        
        config_path = filepath.replace('.pt', '_config.json').replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(save_dict['model_config'], f, indent=2)
        print(f"✓ Config saved to {config_path}")
    
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


class ONNXInferenceEngine:
    """ONNX inference engine with dynamic shape support"""
    
    def __init__(self, onnx_path: str):
        """Initialize ONNX inference engine"""
        import onnxruntime as ort
        
        # Load ONNX model
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Load config
        config_path = onnx_path.replace('.onnx', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        self.behavior_classes = self.config.get('behavior_classes', 
                                                 ['walking', 'standing', 'looking', 'crossing'])
        
        print(f"✓ ONNX Inference Engine initialized")
        print(f"  - Input: {self.input_name}")
        print(f"  - Outputs: {self.output_names}")
    
    def predict(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference with dynamic input shape"""
        # Handle different input shapes
        if trajectory.ndim == 2:
            # Single trajectory: (seq_len, 2) -> (1, seq_len, 2)
            input_data = trajectory.astype(np.float32)[np.newaxis, :, :]
        else:
            input_data = trajectory.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        
        return outputs[0], outputs[1], outputs[2]
    
    def predict_with_labels(self, trajectory: np.ndarray, sequence_id: str = "") -> Stage1Output:
        """Run inference and return formatted output"""
        logits, embedding, probs = self.predict(trajectory)
        
        pred_class_idx = np.argmax(logits[0])
        behavior_label = self.behavior_classes[pred_class_idx]
        
        probs_dict = {
            self.behavior_classes[i]: float(probs[0][i])
            for i in range(len(self.behavior_classes))
        }
        
        return Stage1Output(
            sequence_id=sequence_id,
            behavior_label=behavior_label,
            behavior_embedding=embedding[0].tolist(),
            behavior_probabilities=probs_dict
        )


class Stage1Processor:
    """Stage 1 Data Processor - Main interface for behavior recognition"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', use_onnx: bool = False):
        self.device = device
        self.use_onnx = use_onnx
        self.model = None
        self.onnx_engine = None
        
        if model_path and os.path.exists(model_path):
            if use_onnx and model_path.endswith('.onnx'):
                self.onnx_engine = ONNXInferenceEngine(model_path)
                print(f"Using ONNX runtime: {model_path}")
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                self.model, _ = CNNLSTMBehaviorRecognizer.load_pt(model_path, device)
                self.model.to(device)
                self.model.eval()
                print(f"Using PyTorch model: {model_path}")
            elif model_path.endswith('.onnx') and not use_onnx:
                self.onnx_engine = ONNXInferenceEngine(model_path)
                print(f"Using ONNX model with compatible interface: {model_path}")
    
    def process_trajectory(
        self,
        trajectory: np.ndarray,
        sequence_id: str,
        output_format: str = "dict"
    ) -> Union[Dict, str, List]:
        """Process a single trajectory and return results"""
        # Validate input
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(f"Trajectory must be of shape (seq_len, 2), got {trajectory.shape}")
        
        # Run inference
        if self.onnx_engine:
            output = self.onnx_engine.predict_with_labels(trajectory, sequence_id)
        elif self.model:
            output = self.model.predict(trajectory, sequence_id)
        else:
            raise ValueError("No model loaded. Please provide model_path when initializing processor.")
        
        # Format output
        if output_format == "dict":
            return output.to_dict()
        elif output_format == "json":
            return json.dumps(output.to_dict(), indent=2)
        elif output_format == "csv":
            return output.to_csv_row()
        else:
            return output.to_dict()
    
    def process_batch(
        self,
        trajectories: List[np.ndarray],
        sequence_ids: List[str],
        output_format: str = "dict"
    ) -> List[Union[Dict, str, List]]:
        """Batch process multiple trajectories"""
        results = []
        for traj, seq_id in zip(trajectories, sequence_ids):
            result = self.process_trajectory(traj, seq_id, output_format)
            results.append(result)
        return results
    
    def process_to_dataframe(
        self,
        trajectories: List[np.ndarray],
        sequence_ids: List[str],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Process trajectories and return as DataFrame"""
        results = []
        for traj, seq_id in zip(trajectories, sequence_ids):
            output = self.process_trajectory(traj, seq_id, "dict")
            results.append(output)
        
        df = pd.DataFrame(results)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Results saved to {output_file}")
        
        return df
    
    def get_output_spec(self) -> Dict:
        """Return Stage 1 output format specification"""
        return {
            "stage": "Stage 1 - Behavior Recognition",
            "description": "Extracts behavior features from pedestrian trajectories",
            "input_format": {
                "trajectory": {
                    "type": "np.ndarray",
                    "shape": "(seq_len, 2)",
                    "description": "Sequence of (x, y) coordinates"
                },
                "sequence_id": {
                    "type": "str",
                    "description": "Unique identifier for the trajectory"
                }
            },
            "output_format": {
                "sequence_id": {
                    "type": "str",
                    "description": "Same as input sequence_id"
                },
                "behavior_label": {
                    "type": "str",
                    "description": "Predicted behavior class",
                    "possible_values": ["walking", "standing", "looking", "crossing"]
                },
                "behavior_embedding": {
                    "type": "List[float]",
                    "dimension": 64,
                    "description": "Dense feature vector for downstream tasks"
                },
                "behavior_probabilities": {
                    "type": "Dict[str, float]",
                    "description": "Probability distribution over behavior classes"
                }
            },
            "example": {
                "sequence_id": "video_001_ped_001",
                "behavior_label": "crossing",
                "behavior_embedding": [0.12, -0.34, 0.56],
                "behavior_probabilities": {
                    "walking": 0.05,
                    "standing": 0.02,
                    "looking": 0.08,
                    "crossing": 0.85
                }
            }
        }


# 修复 JSON 序列化问题的自定义编码器
class SafeJSONEncoder(json.JSONEncoder):
    """安全的 JSON 编码器，处理不可序列化的对象"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif obj is ...:
            return None
        else:
            return str(obj)


# Output interface for downstream modules
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
        
        return (f"Pedestrian {stage1_output['sequence_id']} is {top_behavior} "
                f"(confidence: {confidence:.2%})")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1 OUTPUT INTERFACE DEMO")
    print("=" * 60)
    
    # Create dummy trajectory
    seq_len = 30
    sample_trajectory = np.cumsum(np.random.randn(seq_len, 2) * 5, axis=0)
    
    # Method 1: Train a demo model and use PyTorch
    print("\n1. Using PyTorch Model:")
    print("-" * 40)
    
    # Create and train a simple model (for demo)
    model = CNNLSTMBehaviorRecognizer()
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_pt("models/demo_model.pt", metadata={"demo": True})
    
    # Initialize processor with PyTorch
    processor = Stage1Processor(model_path="models/demo_model.pt", use_onnx=False)
    
    # Process trajectory using the MAIN OUTPUT INTERFACE
    result = processor.process_trajectory(
        trajectory=sample_trajectory,
        sequence_id="demo_001",
        output_format="dict"
    )
    
    print(f"Output: {json.dumps(result, indent=2, cls=SafeJSONEncoder)}")
    
    # Method 2: Export to ONNX with dynamic size
    print("\n2. Exporting to ONNX with Dynamic Size:")
    print("-" * 40)
    
    model.save_onnx_dynamic(
        "models/demo_model.onnx",
        input_seq_len=30,
        use_dynamic_batch=True,
        use_dynamic_seq_len=True
    )
    
    # Method 3: Use ONNX runtime for inference
    print("\n3. Using ONNX Runtime (Dynamic Size):")
    print("-" * 40)
    
    processor_onnx = Stage1Processor(model_path="models/demo_model.onnx", use_onnx=True)
    
    # Test with different sequence lengths
    for test_len in [20, 30, 40]:
        test_traj = np.cumsum(np.random.randn(test_len, 2) * 5, axis=0)
        result_onnx = processor_onnx.process_trajectory(
            trajectory=test_traj,
            sequence_id=f"test_len_{test_len}",
            output_format="dict"
        )
        print(f"  Sequence length {test_len}: {result_onnx['behavior_label']} "
              f"(confidence: {max(result_onnx['behavior_probabilities'].values()):.3f})")
    
    # Method 4: Batch processing
    print("\n4. Batch Processing:")
    print("-" * 40)
    
    trajectories = [
        np.cumsum(np.random.randn(30, 2) * 5, axis=0),
        np.cumsum(np.random.randn(30, 2) * 4, axis=0),
        np.cumsum(np.random.randn(30, 2) * 6, axis=0)
    ]
    ids = ["ped_001", "ped_002", "ped_003"]
    
    df = processor.process_to_dataframe(trajectories, ids, "stage1_results.csv")
    print(f"Results DataFrame:\n{df}")
    
    # Method 5: Output spec for downstream (修复 JSON 序列化问题)
    print("\n5. Output Specification for Stage 2:")
    print("-" * 40)
    
    spec = processor.get_output_spec()
    # 使用自定义编码器
    print(json.dumps(spec, indent=2, cls=SafeJSONEncoder))
    
    # Method 6: Convert to downstream format
    print("\n6. Converting to Stage 2 Format:")
    print("-" * 40)
    
    downstream_format = Stage1OutputInterface.to_downstream_format(result)
    print(json.dumps(downstream_format, indent=2, cls=SafeJSONEncoder))
    
    message = Stage1OutputInterface.to_message_format(result)
    print(f"Human-readable: {message}")
    
    print("\n" + "=" * 60)
    print("✓ Stage 1 Output Interface Demo Complete")
    print("=" * 60)
