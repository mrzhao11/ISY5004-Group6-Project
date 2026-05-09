import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Stage1Output:
    """Stage 1 output data format"""
    sequence_id: str
    action_label: str
    look_label: str
    behavior_embedding: List[float]
    action_probabilities: Dict[str, float]
    look_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "action_label": self.action_label,
            "look_label": self.look_label,
            "behavior_embedding": self.behavior_embedding,
            "action_probabilities": self.action_probabilities,
            "look_probabilities": self.look_probabilities
        }
    
    def to_csv_row(self) -> List:
        return [
            self.sequence_id,
            self.action_label,
            self.look_label,
            json.dumps(self.behavior_embedding),
            json.dumps(self.action_probabilities),
            json.dumps(self.look_probabilities)
        ]


class MultiTaskBehaviorRecognizer(nn.Module):
    """Multi-task model for predicting both action and look behavior"""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_action_classes: int = 2,
        num_look_classes: int = 2,
        embedding_dim: int = 64,
        target_length: int = 32
    ):
        super(MultiTaskBehaviorRecognizer, self).__init__()
        
        self.target_length = target_length
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_action_classes)
        )
        
        self.look_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_look_classes)
        )
        
        self.embedding_proj = nn.Linear(128, embedding_dim)
        
        self.action_classes = ['standing', 'walking']
        self.look_classes = ['not-looking', 'looking']
    
    def _adaptive_pool_1d(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        return nn.functional.interpolate(
            x, size=target_length, mode='linear', align_corners=False
        )
    
    def forward(self, trajectory: torch.Tensor):
        batch_size, seq_len, input_dim = trajectory.shape
        
        cnn_input = trajectory.permute(0, 2, 1)
        cnn_features = self.cnn(cnn_input)
        
        current_length = cnn_features.shape[2]
        if current_length != self.target_length:
            cnn_features = self._adaptive_pool_1d(cnn_features, self.target_length)
        
        lstm_input = cnn_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        
        last_hidden = lstm_out[:, -1, :]
        shared_features = self.shared_fc(last_hidden)
        
        action_logits = self.action_classifier(shared_features)
        look_logits = self.look_classifier(shared_features)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        look_probs = torch.softmax(look_logits, dim=-1)
        
        embedding = self.embedding_proj(shared_features)
        
        return action_logits, look_logits, embedding, action_probs, look_probs
    
    def predict(self, trajectory: np.ndarray, sequence_id: str) -> Stage1Output:
        self.eval()
        trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, look_logits, embedding, action_probs, look_probs = self.forward(trajectory_tensor)
        
        action_pred_idx = torch.argmax(action_logits, dim=-1).item()
        look_pred_idx = torch.argmax(look_logits, dim=-1).item()
        
        action_label = self.action_classes[action_pred_idx]
        look_label = self.look_classes[look_pred_idx]
        
        action_probs_dict = {
            self.action_classes[i]: action_probs[0][i].item()
            for i in range(len(self.action_classes))
        }
        
        look_probs_dict = {
            self.look_classes[i]: look_probs[0][i].item()
            for i in range(len(self.look_classes))
        }
        
        return Stage1Output(
            sequence_id=sequence_id,
            action_label=action_label,
            look_label=look_label,
            behavior_embedding=embedding[0].tolist(),
            action_probabilities=action_probs_dict,
            look_probabilities=look_probs_dict
        )
    
    def save_pt(self, filepath: str, metadata: Optional[Dict] = None):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.cnn[0].in_channels,
                'hidden_dim': self.lstm.hidden_size,
                'num_layers': self.lstm.num_layers,
                'num_action_classes': self.action_classifier[-1].out_features,
                'num_look_classes': self.look_classifier[-1].out_features,
                'embedding_dim': self.embedding_proj.out_features,
                'action_classes': self.action_classes,
                'look_classes': self.look_classes,
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
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_action_classes=config['num_action_classes'],
            num_look_classes=config['num_look_classes'],
            embedding_dim=config['embedding_dim'],
            target_length=config.get('target_length', 32)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', None)
        print(f"✓ Model loaded from {filepath}")
        return model, metadata


def calculate_metrics(y_true, y_pred, y_prob, class_names):
    metrics = {}
    
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if len(class_names) == 2 and y_prob is not None:
        positive_idx = 1
        y_prob_positive = y_prob[:, positive_idx]
        try:
            metrics['roc_auc'] = roc_auc_score(y_true == class_names[positive_idx], y_prob_positive)
        except:
            metrics['roc_auc'] = 0.0
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_name).astype(int)
        y_pred_binary = (y_pred == class_name).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if y_prob is not None and len(y_prob) > 0 and i < y_prob.shape[1]:
            ap = average_precision_score(y_true_binary, y_prob[:, i])
        else:
            ap = 0.0
        
        metrics['per_class'][class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': ap,
            'support': int(np.sum(y_true_binary))
        }
    
    if y_prob is not None and y_prob.size > 0:
        metrics['mAP'] = np.mean([metrics['per_class'][c]['average_precision'] for c in class_names])
    else:
        metrics['mAP'] = 0.0
    
    return metrics


def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    
    all_action_preds = []
    all_action_labels = []
    all_action_probs = []
    all_look_preds = []
    all_look_labels = []
    all_look_probs = []
    
    with torch.no_grad():
        for batch_x, batch_action_y, batch_look_y in tqdm(val_loader, desc="  Evaluating", leave=False):
            batch_x = batch_x.to(device)
            batch_action_y = batch_action_y.to(device)
            batch_look_y = batch_look_y.to(device)
            
            action_logits, look_logits, _, action_probs, look_probs = model(batch_x)
            
            _, action_predicted = torch.max(action_logits, 1)
            _, look_predicted = torch.max(look_logits, 1)
            
            all_action_preds.extend(action_predicted.cpu().numpy())
            all_action_labels.extend(batch_action_y.cpu().numpy())
            all_action_probs.extend(action_probs.cpu().numpy())
            
            all_look_preds.extend(look_predicted.cpu().numpy())
            all_look_labels.extend(batch_look_y.cpu().numpy())
            all_look_probs.extend(look_probs.cpu().numpy())
    
    all_action_preds = np.array(all_action_preds)
    all_action_labels = np.array(all_action_labels)
    all_action_probs = np.array(all_action_probs) if len(all_action_probs) > 0 else None
    
    all_look_preds = np.array(all_look_preds)
    all_look_labels = np.array(all_look_labels)
    all_look_probs = np.array(all_look_probs) if len(all_look_probs) > 0 else None
    
    action_label_to_name = {i: name for i, name in enumerate(model.action_classes)}
    look_label_to_name = {i: name for i, name in enumerate(model.look_classes)}
    
    y_true_action_names = [action_label_to_name[label] for label in all_action_labels]
    y_pred_action_names = [action_label_to_name[pred] for pred in all_action_preds]
    
    y_true_look_names = [look_label_to_name[label] for label in all_look_labels]
    y_pred_look_names = [look_label_to_name[pred] for pred in all_look_preds]
    
    action_metrics = calculate_metrics(y_true_action_names, y_pred_action_names, all_action_probs, model.action_classes)
    look_metrics = calculate_metrics(y_true_look_names, y_pred_look_names, all_look_probs, model.look_classes)
    
    print(f"\n  Action Classification Confusion Matrix:")
    cm_action = confusion_matrix(all_action_labels, all_action_preds, labels=range(len(model.action_classes)))
    print("  " + " " * 15 + "".join([f"{model.action_classes[j][:10]:>12}" for j in range(cm_action.shape[1])]))
    for i in range(cm_action.shape[0]):
        print(f"  {model.action_classes[i][:12]:<12} " + "".join([f"{cm_action[i][j]:>12d}" for j in range(cm_action.shape[1])]))
    
    print(f"\n  Look Classification Confusion Matrix:")
    cm_look = confusion_matrix(all_look_labels, all_look_preds, labels=range(len(model.look_classes)))
    print("  " + " " * 15 + "".join([f"{model.look_classes[j][:10]:>12}" for j in range(cm_look.shape[1])]))
    for i in range(cm_look.shape[0]):
        print(f"  {model.look_classes[i][:12]:<12} " + "".join([f"{cm_look[i][j]:>12d}" for j in range(cm_look.shape[1])]))
    
    return action_metrics, look_metrics


def print_metrics_table(action_metrics, look_metrics, model, title="Evaluation Metrics"):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    print(f"\n📊 ACTION CLASSIFICATION (Walking vs Standing):")
    print("-" * 40)
    print(f"  Accuracy:           {action_metrics['accuracy']:.4f} ({action_metrics['accuracy']*100:.2f}%)")
    print(f"  Weighted Precision: {action_metrics['precision']:.4f}")
    print(f"  Weighted Recall:    {action_metrics['recall']:.4f}")
    print(f"  Weighted F1-Score:  {action_metrics['f1_weighted']:.4f}")
    print(f"  Macro F1-Score:     {action_metrics['f1_macro']:.4f}")
    if 'roc_auc' in action_metrics:
        print(f"  ROC-AUC:            {action_metrics['roc_auc']:.4f}")
    
    print(f"\n  Per-Class Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    for class_name in model.action_classes:
        class_metrics = action_metrics['per_class'][class_name]
        print(f"  {class_name:<12} {class_metrics['precision']:.4f}     {class_metrics['recall']:.4f}     {class_metrics['f1_score']:.4f}     {class_metrics['support']:<10}")
    
    print(f"\n📊 LOOK CLASSIFICATION (Looking vs Not-looking):")
    print("-" * 40)
    print(f"  Accuracy:           {look_metrics['accuracy']:.4f} ({look_metrics['accuracy']*100:.2f}%)")
    print(f"  Weighted Precision: {look_metrics['precision']:.4f}")
    print(f"  Weighted Recall:    {look_metrics['recall']:.4f}")
    print(f"  Weighted F1-Score:  {look_metrics['f1_weighted']:.4f}")
    print(f"  Macro F1-Score:     {look_metrics['f1_macro']:.4f}")
    if 'roc_auc' in look_metrics:
        print(f"  ROC-AUC:            {look_metrics['roc_auc']:.4f}")
    
    print(f"\n  Per-Class Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    for class_name in model.look_classes:
        class_metrics = look_metrics['per_class'][class_name]
        print(f"  {class_name:<12} {class_metrics['precision']:.4f}     {class_metrics['recall']:.4f}     {class_metrics['f1_score']:.4f}     {class_metrics['support']:<10}")
    
    print("=" * 80)


class JAADDataLoader:
    """JAAD dataset loader for multi-task learning"""
    
    def __init__(self, metadata_path: str, manifest_path: str):
        self.metadata_path = Path(metadata_path)
        self.manifest_path = Path(manifest_path)
        
        self.metadata = pd.read_csv(self.metadata_path)
        self.sequences_manifest = pd.read_csv(self.manifest_path)
        
        print(f"✓ Loaded dataset")
        print(f"  - Metadata: {len(self.metadata)} rows")
        print(f"  - Sequences manifest: {len(self.sequences_manifest)} rows")
        
        self._build_trajectory_cache()
    
    def _build_trajectory_cache(self):
        print("\n  Building trajectory cache...")
        
        self.trajectory_cache = {}
        self.action_label_cache = {}
        self.look_label_cache = {}
        
        metadata_groups = {}
        for (video_id, ped_id), group in tqdm(self.metadata.groupby(['video_id', 'pedestrian_id']), desc="  Processing metadata"):
            key = f"{video_id}__{ped_id}"
            group_sorted = group.sort_values('frame_id')
            centers_x = group_sorted['xtl'] + group_sorted['width'] / 2
            centers_y = group_sorted['ytl'] + group_sorted['height'] / 2
            trajectory = np.column_stack([centers_x.values, centers_y.values]).astype(np.float32)
            metadata_groups[key] = trajectory
        
        print(f"  Created {len(metadata_groups)} trajectory groups")
        
        for idx, row in tqdm(self.sequences_manifest.iterrows(), desc="  Processing sequences", total=len(self.sequences_manifest)):
            seq_id = row['sequence_id']
            video_id = row['video_id']
            pedestrian_id = row['pedestrian_id']
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            
            key = f"{video_id}__{pedestrian_id}"
            if key in metadata_groups:
                full_traj = metadata_groups[key]
                if len(full_traj) >= 10:
                    self.trajectory_cache[seq_id] = full_traj
                    self.action_label_cache[seq_id] = self._get_action_label(row['sequence_behavior_label'])
                    self.look_label_cache[seq_id] = self._get_look_label_for_sequence(video_id, pedestrian_id, start_frame, end_frame)
        
        print(f"  ✓ Built cache with {len(self.trajectory_cache)} sequences")
        
        action_counts = {}
        look_counts = {}
        for seq_id in self.trajectory_cache.keys():
            action = self.action_label_cache.get(seq_id, 'unknown')
            look = self.look_label_cache.get(seq_id, 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
            look_counts[look] = look_counts.get(look, 0) + 1
        
        print(f"  Action distribution: {action_counts}")
        print(f"  Look distribution: {look_counts}")
    
    def _get_action_label(self, behavior_label: str) -> str:
        label_lower = str(behavior_label).lower()
        
        walking_keywords = ['walk', 'walking', 'moving', 'cross', 'crossing']
        for kw in walking_keywords:
            if kw in label_lower:
                return 'walking'
        
        standing_keywords = ['stand', 'standing', 'wait', 'waiting', 'stop', 'stopped', 'idle', 'static', 'still']
        for kw in standing_keywords:
            if kw in label_lower:
                return 'standing'
        
        return 'walking'
    
    def _get_look_label_for_sequence(self, video_id: str, pedestrian_id: str, start_frame: int, end_frame: int) -> str:
        mask = (self.metadata['video_id'] == video_id) & \
               (self.metadata['pedestrian_id'] == pedestrian_id) & \
               (self.metadata['frame_id'] >= start_frame) & \
               (self.metadata['frame_id'] <= end_frame)
        
        subset = self.metadata[mask]
        
        if len(subset) == 0:
            return 'not-looking'
        
        looking_count = (subset['look_label'] == 'looking').sum()
        not_looking_count = (subset['look_label'] == 'not-looking').sum()
        
        return 'looking' if looking_count > not_looking_count else 'not-looking'
    
    def load_all_trajectories_with_labels(self, min_length: int = 10, sample_size: Optional[int] = None):
        trajectories = []
        action_labels = []
        look_labels = []
        
        for seq_id, traj in tqdm(self.trajectory_cache.items(), desc="  Loading trajectories"):
            if len(traj) < min_length:
                continue
            
            action_label = self.action_label_cache.get(seq_id)
            look_label = self.look_label_cache.get(seq_id)
            
            if action_label and look_label:
                trajectories.append(traj)
                action_labels.append(action_label)
                look_labels.append(look_label)
        
        if sample_size and sample_size < len(trajectories):
            indices = np.random.choice(len(trajectories), sample_size, replace=False)
            trajectories = [trajectories[i] for i in indices]
            action_labels = [action_labels[i] for i in indices]
            look_labels = [look_labels[i] for i in indices]
            print(f"  ✓ Sampled {sample_size} trajectories")
        
        print(f"✓ Loaded {len(trajectories)} labeled trajectories")
        return trajectories, action_labels, look_labels
    
    def load_all_trajectories(self, min_length: int = 10, max_length: int = None):
        """Load all trajectories for inference (for Stage 2)"""
        trajectories = []
        valid_ids = []
        valid_labels = []
        
        for seq_id, traj in tqdm(self.trajectory_cache.items(), desc="  Loading trajectories for inference"):
            if len(traj) < min_length:
                continue
            
            if max_length and len(traj) > max_length:
                traj = traj[:max_length]
            
            action_label = self.action_label_cache.get(seq_id, 'unknown')
            look_label = self.look_label_cache.get(seq_id, 'unknown')
            
            trajectories.append(traj)
            valid_ids.append(seq_id)
            valid_labels.append(f"{action_label}_{look_label}")
        
        print(f"✓ Loaded {len(trajectories)} trajectories for inference")
        return trajectories, valid_ids, valid_labels


def prepare_data_for_training(trajectories, action_labels, look_labels, target_length=32, val_split=0.2, batch_size=32):
    print("\n  Preparing data...")
    
    processed_trajectories = []
    for traj in tqdm(trajectories, desc="  Processing trajectories"):
        if len(traj) > target_length:
            processed_traj = traj[:target_length]
        else:
            pad_length = target_length - len(traj)
            processed_traj = np.vstack([traj, np.zeros((pad_length, 2))])
        processed_trajectories.append(processed_traj)
    
    X = np.array(processed_trajectories)
    
    action_encoder = LabelEncoder()
    look_encoder = LabelEncoder()
    
    y_action = action_encoder.fit_transform(action_labels)
    y_look = look_encoder.fit_transform(look_labels)
    
    print(f"\nAction label encoding: {dict(zip(action_encoder.classes_, range(len(action_encoder.classes_))))}")
    print(f"Look label encoding: {dict(zip(look_encoder.classes_, range(len(look_encoder.classes_))))}")
    
    X_tensor = torch.FloatTensor(X)
    y_action_tensor = torch.LongTensor(y_action)
    y_look_tensor = torch.LongTensor(y_look)
    
    dataset = TensorDataset(X_tensor, y_action_tensor, y_look_tensor)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData split: Train={train_size}, Val={val_size}, Batch size={batch_size}")
    
    return train_loader, val_loader, action_encoder, look_encoder


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    action_criterion = nn.CrossEntropyLoss()
    look_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_action_metrics = None
    best_look_metrics = None
    
    print("\n" + "=" * 60)
    print("Starting Multi-Task Training")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_action_correct = 0
        train_look_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for batch_x, batch_action_y, batch_look_y in train_pbar:
            batch_x = batch_x.to(device)
            batch_action_y = batch_action_y.to(device)
            batch_look_y = batch_look_y.to(device)
            
            optimizer.zero_grad()
            action_logits, look_logits, _, _, _ = model(batch_x)
            
            action_loss = action_criterion(action_logits, batch_action_y)
            look_loss = look_criterion(look_logits, batch_look_y)
            loss = action_loss + look_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, action_pred = torch.max(action_logits, 1)
            _, look_pred = torch.max(look_logits, 1)
            
            train_total += batch_x.size(0)
            train_action_correct += (action_pred == batch_action_y).sum().item()
            train_look_correct += (look_pred == batch_look_y).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_action_acc = train_action_correct / train_total
        train_look_acc = train_look_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_action_correct = 0
        val_look_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch_x, batch_action_y, batch_look_y in val_pbar:
                batch_x = batch_x.to(device)
                batch_action_y = batch_action_y.to(device)
                batch_look_y = batch_look_y.to(device)
                
                action_logits, look_logits, _, _, _ = model(batch_x)
                
                action_loss = action_criterion(action_logits, batch_action_y)
                look_loss = look_criterion(look_logits, batch_look_y)
                loss = action_loss + look_loss
                
                val_loss += loss.item()
                _, action_pred = torch.max(action_logits, 1)
                _, look_pred = torch.max(look_logits, 1)
                
                val_total += batch_x.size(0)
                val_action_correct += (action_pred == batch_action_y).sum().item()
                val_look_correct += (look_pred == batch_look_y).sum().item()
        
        val_action_acc = val_action_correct / val_total
        val_look_acc = val_look_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Action Acc: {train_action_acc:.4f}, Train Look Acc: {train_look_acc:.4f}")
        print(f"  Val Action Acc: {val_action_acc:.4f}, Val Look Acc: {val_look_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  Evaluating best model...")
            best_action_metrics, best_look_metrics = evaluate_model(model, val_loader, device)
            torch.save(model.state_dict(), 'best_multitask_model.pt')
            print(f"  ✓ Best model saved")
        print("-" * 60)
    
    print("\n✓ Training completed!")
    model.load_state_dict(torch.load('best_multitask_model.pt'))
    return model, best_action_metrics, best_look_metrics


class Stage1Processor:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.model, _ = MultiTaskBehaviorRecognizer.load_pt(model_path, device)
            self.model.to(device)
            self.model.eval()
            print(f"Using model: {model_path}")
    
    def process_dataset(self, data_loader: JAADDataLoader, min_length: int = 10, output_file: Optional[str] = None):
        print("\n" + "=" * 60)
        print("Processing Dataset for Stage 2")
        print("=" * 60)
        
        trajectories, sequence_ids, _ = data_loader.load_all_trajectories(min_length=min_length)
        
        if len(trajectories) == 0:
            print("No valid trajectories found!")
            return pd.DataFrame()
        
        print(f"\nProcessing {len(trajectories)} trajectories...")
        results = []
        for traj, seq_id in tqdm(zip(trajectories, sequence_ids), desc="  Processing", total=len(trajectories)):
            if self.model:
                result = self.model.predict(traj, seq_id)
                results.append(result)
        
        print("\nCreating Stage 2 CSV...")
        rows = []
        for result in results:
            row = {'sequence_id': result.sequence_id}
            
            for j, val in enumerate(result.behavior_embedding):
                row[f'beh_emb_{j}'] = val
            
            row['p_walk'] = result.action_probabilities.get('walking', 0.0)
            row['p_stand'] = result.action_probabilities.get('standing', 0.0)
            row['p_look'] = result.look_probabilities.get('looking', 0.0)
            row['p_notlook'] = result.look_probabilities.get('not-looking', 0.0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        for j in range(64):
            if f'beh_emb_{j}' not in df.columns:
                df[f'beh_emb_{j}'] = 0.0
        
        embedding_cols = [f'beh_emb_{j}' for j in range(64)]
        other_cols = ['sequence_id', 'p_walk', 'p_stand', 'p_look', 'p_notlook']
        df = df[embedding_cols + other_cols]
        
        print(f"\n✓ Processing complete! {len(df)} trajectories processed")
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Stage 2 CSV saved to: {output_file}")
        
        return df


def main():
    print("=" * 60)
    print("STAGE 1: MULTI-TASK BEHAVIOR RECOGNITION")
    print("Predicting: Action (Walking/Standing) + Look (Looking/Not-looking)")
    print("=" * 60)
    
    QUICK_TEST = False
    TEST_SAMPLE_SIZE = 5000
    EPOCHS = 50
    
    base_dir = r"E:\有记录\ISY5004-Group6-Project-main (1)\ISY5004-Group6-Project-main"
    metadata_path = os.path.join(base_dir, "data", "filtered_metadata.csv")
    manifest_path = os.path.join(base_dir, "data", "filtered_sequences_manifest.csv")
    
    if not os.path.exists(metadata_path) or not os.path.exists(manifest_path):
        print("Data files not found!")
        return
    
    print("\n✓ Data files found!")
    
    data_loader = JAADDataLoader(metadata_path, manifest_path)
    
    sample_size = TEST_SAMPLE_SIZE if QUICK_TEST else None
    trajectories, action_labels, look_labels = data_loader.load_all_trajectories_with_labels(
        min_length=10, sample_size=sample_size
    )
    
    if len(trajectories) == 0:
        print("No labeled trajectories found!")
        return
    
    if QUICK_TEST:
        EPOCHS = min(EPOCHS, 10)
        print(f"\n⚠ QUICK TEST MODE: {EPOCHS} epochs")
    
    train_loader, val_loader, action_encoder, look_encoder = prepare_data_for_training(
        trajectories, action_labels, look_labels, target_length=32, val_split=0.2, batch_size=32
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠ Training on CPU (slow). Set QUICK_TEST=True for faster testing.")
    
    model = MultiTaskBehaviorRecognizer(num_action_classes=2, num_look_classes=2)
    model.to(device)
    
    print(f"\nModel on {device}")
    print(f"Action classes: {model.action_classes}")
    print(f"Look classes: {model.look_classes}")
    
    model, action_metrics, look_metrics = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=0.001, device=device
    )
    
    print_metrics_table(action_metrics, look_metrics, model, "Final Model Evaluation")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / "stage1_multitask_model.pt")
    
    model.save_pt(model_path, metadata={
        "task": "multi-task",
        "action_classes": model.action_classes,
        "look_classes": model.look_classes,
        "action_accuracy": float(action_metrics['accuracy']),
        "look_accuracy": float(look_metrics['accuracy']),
        "quick_test": QUICK_TEST
    })
    
    processor = Stage1Processor(model_path=model_path, device=device)
    output_csv = "stage2_input.csv"
    df = processor.process_dataset(data_loader, min_length=10, output_file=output_csv)
    
    if len(df) > 0:
        print("\n" + "=" * 60)
        print("STAGE 2 CSV PREVIEW")
        print("=" * 60)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns[:5])} ... {list(df.columns[-5:])}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
        print(f"\n✓ Ready for Stage 2! CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
