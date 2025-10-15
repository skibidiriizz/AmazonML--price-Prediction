#!/usr/bin/env python3
"""
Optimized Wide MLP Training Script
=================================

Streamlined training script with only the optimal wide architecture
for maximum price prediction performance using CLIP embeddings.

Features:
- Fixed wide architecture (6 layers × 2048 neurons)
- CUDA GPU acceleration 
- Advanced regularization (dropout, batch norm, weight decay)
- Learning rate scheduling and early stopping
- Full dataset SMAPE evaluation
- Performance visualizations

Usage:
    python wide_mlp_trainer.py [--epochs 300] [--batch-size 256]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import time
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

class WideMLPOptimal(nn.Module):
    """
    Optimal Wide MLP with 6 layers × 2048 neurons each
    """
    
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=6, dropout_rate=0.3):
        super(WideMLPOptimal, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Build the wide network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers (all same size for wide architecture)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            # Alternate activation functions for variety
            if i % 2 == 0:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze()
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'architecture': 'wide'
        }

class WideMLPTrainer:
    """Optimized trainer for wide MLP"""
    
    def __init__(self, model, device, learning_rate=0.0008, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimizer optimized for wide networks
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7, 
            patience=12,  # Slightly more patience for wide networks
            min_lr=1e-6,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_smape': [],
            'val_smape': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        train_smape = smape(np.array(all_targets), np.array(all_preds))
        
        return avg_loss, train_smape
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        val_smape = smape(np.array(all_targets), np.array(all_preds))
        
        return avg_loss, val_smape
    
    def train(self, train_loader, val_loader, epochs=300, early_stopping_patience=25):
        """Train the model with early stopping"""
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"📊 Model: {self.model.get_model_info()['total_parameters']:,} parameters")
        print(f"🎮 Device: {self.device}")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epoch_pbar = tqdm(range(epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            start_time = time.time()
            
            # Train and validate
            train_loss, train_smape = self.train_epoch(train_loader)
            val_loss, val_smape = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_smape'].append(train_smape)
            self.history['val_smape'].append(val_smape)
            self.history['learning_rates'].append(current_lr)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update progress bar
            epoch_time = time.time() - start_time
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train SMAPE': f'{train_smape:.2f}%',
                'Val SMAPE': f'{val_smape:.2f}%',
                'LR': f'{current_lr:.2e}',
                'Time': f'{epoch_time:.1f}s'
            })
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Print detailed progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"\\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Train - Loss: {train_loss:.4f}, SMAPE: {train_smape:.2f}%")
                print(f"  Val   - Loss: {val_loss:.4f}, SMAPE: {val_smape:.2f}%")
                print(f"  Learning Rate: {current_lr:.2e}")
                print(f"  Best Val Loss: {best_val_loss:.4f} (Patience: {patience_counter}/{early_stopping_patience})")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return self.history

def load_embeddings_and_targets():
    """Load embeddings and target values"""
    print("📂 Loading embeddings and target data...")
    
    # Load embeddings (using aligned embeddings)
    embeddings_dir = Path('embeddings_aligned')
    text_embeddings = np.load(embeddings_dir / 'text_embeddings_normalized.npy')
    image_embeddings = np.load(embeddings_dir / 'image_embeddings_normalized.npy')
    
    print(f"✓ Text embeddings loaded: {text_embeddings.shape}")
    print(f"✓ Image embeddings loaded: {image_embeddings.shape}")
    
    # Load metadata
    with open(embeddings_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    sample_ids = metadata['sample_ids']
    print(f"✓ Sample IDs loaded: {len(sample_ids)} samples")
    
    # Load target values (using aligned data)
    output_csv = pd.read_csv('dataset/train_final_out_aligned.csv')
    print(f"✓ Target data loaded: {len(output_csv)} prices (aligned)")
    
    # Match embeddings with target values
    embedding_df = pd.DataFrame({
        'sample_id': sample_ids,
        'text_emb_idx': range(len(sample_ids))
    })
    
    merged_df = embedding_df.merge(output_csv, on='sample_id', how='inner')
    print(f"✓ Matched samples: {len(merged_df)}")
    
    # Extract matched embeddings and targets
    matched_indices = merged_df['text_emb_idx'].values
    X_text = text_embeddings[matched_indices]
    X_image = image_embeddings[matched_indices]
    y = merged_df['price'].values
    matched_sample_ids = merged_df['sample_id'].values
    
    # Fuse embeddings via concatenation
    X_combined = np.concatenate([X_text, X_image], axis=1)
    print(f"✓ Combined features shape: {X_combined.shape}")
    
    print(f"\\n📊 Target statistics:")
    print(f"   - Min price: ${y.min():.2f}")
    print(f"   - Max price: ${y.max():.2f}")
    print(f"   - Mean price: ${y.mean():.2f}")
    print(f"   - Std price: ${y.std():.2f}")
    
    return X_combined, y, matched_sample_ids

def create_data_loaders(X, y, batch_size=256, val_split=0.2, random_state=42):
    """Create PyTorch data loaders"""
    print(f"\\n🔄 Creating data loaders...")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=None
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, scaler

def evaluate_on_full_dataset(model, X, y, scaler, device, batch_size=512):
    """Evaluate model on the entire dataset and calculate SMAPE"""
    print(f"\\n🔍 Evaluating on full dataset ({X.shape[0]} samples)...")
    
    model.eval()
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and loader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = total_loss / len(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    smape_score = smape(all_targets, all_preds)
    
    # Calculate R-squared
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\\n📊 FULL DATASET EVALUATION RESULTS:")
    print("=" * 50)
    print(f"🎯 SMAPE:       {smape_score:.3f}%")
    print(f"📈 RMSE:        ${rmse:.2f}")
    print(f"📊 MAE:         ${mae:.2f}")
    print(f"🔢 MSE:         {mse:.4f}")
    print(f"📈 R²:          {r2:.4f}")
    print(f"📊 Samples:     {len(all_targets):,}")
    print("=" * 50)
    
    return {
        'smape': smape_score,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }

def save_results(model, trainer, evaluation_results, args, save_dir):
    """Save all results and model"""
    print(f"\\n💾 Saving results to {save_dir}...")
    
    # Model info
    model_info = model.get_model_info()
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model_info,
        'training_args': vars(args),
        'evaluation_results': evaluation_results,
        'training_history': trainer.history
    }, save_dir / 'wide_mlp_optimal.pth')
    
    # Save evaluation results
    results_df = pd.DataFrame([evaluation_results])
    results_df.to_csv(save_dir / 'evaluation_results.csv', index=False)
    
    # Save training history
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(save_dir / 'training_history.csv', index=False)
    
    print("✓ All results saved")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Optimized Wide MLP Training')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers (default: 6)')
    parser.add_argument('--hidden-size', type=int, default=2048, help='Hidden layer size (default: 2048)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum epochs (default: 300)')
    parser.add_argument('--learning-rate', type=float, default=0.0008, help='Learning rate (default: 0.0008)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience (default: 25)')
    
    args = parser.parse_args()
    
    print("🚀 OPTIMIZED WIDE MLP TRAINING")
    print("=" * 60)
    print(f"Architecture: {args.layers} layers × {args.hidden_size} neurons")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load data
        X, y, sample_ids = load_embeddings_and_targets()
        input_size = X.shape[1]
        
        # Create data loaders
        train_loader, val_loader, scaler = create_data_loaders(
            X, y, batch_size=args.batch_size, val_split=0.2
        )
        
        # Create model
        model = WideMLPOptimal(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout_rate=args.dropout
        )
        
        model_info = model.get_model_info()
        print(f"\\n📊 Model Statistics:")
        print(f"   Architecture: {args.layers} layers × {args.hidden_size} neurons")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Model size: ~{model_info['total_parameters'] * 4 / 1e6:.1f} MB")
        
        # Create trainer
        trainer = WideMLPTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        print(f"\\n🎯 Training Configuration:")
        print(f"   Max epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Dropout: {args.dropout}")
        print(f"   Early stopping patience: {args.patience}")
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping_patience=args.patience
        )
        
        # Evaluate on full dataset
        evaluation_results = evaluate_on_full_dataset(
            model=model,
            X=X,
            y=y,
            scaler=scaler,
            device=device,
            batch_size=args.batch_size
        )
        
        # Create results directory
        results_dir = Path('wide_mlp_optimal_results')
        results_dir.mkdir(exist_ok=True)
        
        # Save results
        save_results(model, trainer, evaluation_results, args, results_dir)
        
        # Final summary
        print(f"\\n✅ OPTIMIZED WIDE MLP TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🎯 Final SMAPE: {evaluation_results['smape']:.3f}%")
        print(f"📈 RMSE: ${evaluation_results['rmse']:.2f}")
        print(f"📊 MAE: ${evaluation_results['mae']:.2f}")
        print(f"📈 R²: {evaluation_results['r2']:.4f}")
        print(f"🔢 Parameters: {model_info['total_parameters']:,}")
        print(f"📁 Results saved to: {results_dir}/")
        print("=" * 60)
        
        # Performance assessment
        smape_score = evaluation_results['smape']
        if smape_score < 15:
            print("🏆 OUTSTANDING: SMAPE < 15% - Excellent performance!")
        elif smape_score < 20:
            print("✅ EXCELLENT: SMAPE < 20% - Very good performance!")
        elif smape_score < 25:
            print("📈 GOOD: SMAPE < 25% - Solid performance!")
        else:
            print("⚠️  FAIR: SMAPE > 25% - Room for improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)