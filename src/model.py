import os
# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json
import logging
from config import Config, LSTMConfig, AttentionConfig, TrainingConfig

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def save_model_safely(model, train_losses, val_losses, total_params, trainable_params, output_dir, Config):
    """Safe model saving that avoids pickle issues"""
    import numpy as np

    # Convert Config to dict safely, avoiding mappingproxy objects
    config_dict = {}
    for key in dir(Config):
        if not key.startswith('_') and not callable(getattr(Config, key)):
            try:
                value = getattr(Config, key)
                # Convert to basic types to avoid pickle issues
                if isinstance(value, (int, float, str, bool, list, tuple)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
            except:
                pass

    # Save final model with safe data types
    final_model_path = os.path.join(output_dir, 'models', 'final_model.pth')

    save_dict = {
        'model_state_dict': model.state_dict(),
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': float(train_losses[-1]),
            'best_val_loss': float(min(val_losses)),
            'best_epoch': int(np.argmin(val_losses) + 1),
            'total_epochs': len(train_losses)
        },
        'model_info': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_type': 'EnhancedLSTM',
            'architecture': 'Enhanced LSTM with Attention, Multi-Scale CNN, Peak Detection'
        },
        'config': config_dict
    }

    torch.save(save_dict, final_model_path)
    print(f"Model saved successfully to: {final_model_path}")

    return final_model_path


class MilanTrafficDataset(Dataset):
    """Enhanced Dataset class with data augmentation"""

    def __init__(self, X_traffic, X_temporal, y, augment=False, config=None):
        self.X_traffic = torch.FloatTensor(X_traffic)
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y = torch.FloatTensor(y)
        self.augment = augment and config.USE_DATA_AUGMENTATION if config else False
        self.noise_factor = config.NOISE_FACTOR if config else 0.01

    def __len__(self):
        return len(self.X_traffic)

    def __getitem__(self, idx):
        x_traffic = self.X_traffic[idx]
        x_temporal = self.X_temporal[idx]
        y = self.y[idx]

        if self.augment and np.random.random() > 0.5:
            # Add small amount of noise
            noise = torch.randn_like(x_traffic) * self.noise_factor
            x_traffic = x_traffic + noise

        return x_traffic, x_temporal, y


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like attention"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for capturing different temporal patterns"""

    def __init__(self, input_dim, target_output_dim, kernel_sizes=[3, 5, 7]):
        super(MultiScaleCNN, self).__init__()

        # Calculate exact output dimensions to avoid any mismatch
        self.n_convs = len(kernel_sizes)
        base_dim = target_output_dim // self.n_convs
        remainder = target_output_dim % self.n_convs

        self.convs = nn.ModuleList()
        self.conv_dims = []

        for i, k in enumerate(kernel_sizes):
            # Distribute remainder across first few convolutions
            current_dim = base_dim + (1 if i < remainder else 0)
            self.convs.append(
                nn.Conv1d(input_dim, current_dim, kernel_size=k, padding=k // 2)
            )
            self.conv_dims.append(current_dim)

        # Calculate actual total output dimension
        self.actual_output_dim = sum(self.conv_dims)

        # Use the exact calculated dimension for BatchNorm
        self.bn = nn.BatchNorm1d(self.actual_output_dim)
        self.dropout = nn.Dropout(0.1)

        print(f"MultiScaleCNN: target={target_output_dim}, actual={self.actual_output_dim}, dims={self.conv_dims}")

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        output = torch.cat(conv_outputs, dim=1)
        output = self.bn(output)
        output = self.dropout(output)

        return output.transpose(1, 2)  # Back to (batch, seq_len, features)


class EnhancedLSTMTrafficPredictor(nn.Module):
    """Enhanced LSTM with attention, residual connections, and peak prediction focus"""

    def __init__(self, n_squares=30, n_temporal_features=10, config=None):
        super(EnhancedLSTMTrafficPredictor, self).__init__()

        self.config = config or Config
        self.n_squares = n_squares
        self.n_temporal_features = n_temporal_features
        self.hidden_dim = self.config.HIDDEN_DIM
        self.num_layers = self.config.NUM_LAYERS
        self.pred_horizon = self.config.PREDICTION_HORIZON

        print(f"Initializing Enhanced LSTM with:")
        print(f"  n_squares: {n_squares}")
        print(f"  n_temporal_features: {n_temporal_features}")
        print(f"  hidden_dim: {self.hidden_dim}")

        # Multi-scale CNN preprocessing with proper dimension handling
        if self.config.USE_MULTI_SCALE:
            cnn_target_dim = self.hidden_dim // 2
            self.traffic_cnn = MultiScaleCNN(
                n_squares, cnn_target_dim, self.config.KERNEL_SIZES
            )
            # Use the actual output dimension from CNN
            lstm_input_size = self.traffic_cnn.actual_output_dim
            print(f"  CNN output dim: {lstm_input_size}")
        else:
            self.traffic_cnn = None
            lstm_input_size = n_squares
            print(f"  Direct LSTM input: {lstm_input_size}")

        # Main LSTM layers
        self.traffic_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.config.DROPOUT if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Temporal feature processing - FIXED DIMENSIONS
        temporal_hidden_dim = self.hidden_dim // 2
        print(f"  Temporal encoder dims: {n_temporal_features} -> {temporal_hidden_dim} -> {temporal_hidden_dim}")

        # Create temporal encoder without problematic BatchNorm on sequence data
        if self.config.USE_BATCH_NORM:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(n_temporal_features, temporal_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT),
                nn.Linear(temporal_hidden_dim, temporal_hidden_dim)
            )
            # Separate BatchNorm for proper application
            self.temporal_bn = nn.BatchNorm1d(temporal_hidden_dim)
        else:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(n_temporal_features, temporal_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT),
                nn.Linear(temporal_hidden_dim, temporal_hidden_dim)
            )
            self.temporal_bn = None

        # Combine traffic and temporal
        fusion_input_dim = self.hidden_dim * 2 + temporal_hidden_dim
        fusion_output_dim = self.hidden_dim * 2
        print(f"  Fusion dims: {fusion_input_dim} -> {fusion_output_dim}")

        if self.config.USE_LAYER_NORM:
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_output_dim),
                nn.ReLU(),
                nn.LayerNorm(fusion_output_dim),
                nn.Dropout(self.config.DROPOUT)
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_output_dim),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT)
            )

        # Enhanced attention mechanism
        if self.config.USE_ATTENTION:
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_output_dim,
                num_heads=8,
                dropout=self.config.DROPOUT,
                batch_first=True
            )

            if AttentionConfig.USE_POSITIONAL_ENCODING:
                self.pos_encoding = PositionalEncoding(fusion_output_dim)

        # Peak-aware prediction heads
        self.prediction_heads = nn.ModuleList([
            self._create_prediction_head() for _ in range(self.pred_horizon)
        ])

        # Peak detection - CHANGED TO LOGITS for autocast compatibility
        self.peak_detector = nn.Sequential(
            nn.Linear(fusion_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
            # Removed Sigmoid - will use logits
        )

        # Residual connections
        if self.config.USE_RESIDUAL_CONNECTIONS:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
            self.peak_residual_weight = nn.Parameter(torch.tensor(0.3))

        # Peak enhancement layer
        self.peak_enhancer = nn.Sequential(
            nn.Linear(n_squares, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_squares),
            nn.Tanh()
        )

        print("Enhanced LSTM initialization completed successfully!")

    def _create_prediction_head(self):
        """Create a single prediction head with better capacity"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_squares)
        )

    def forward(self, x_traffic, x_temporal):
        batch_size, seq_len, _ = x_traffic.shape

        # Multi-scale CNN preprocessing
        if self.traffic_cnn is not None:
            x_traffic_processed = self.traffic_cnn(x_traffic)
        else:
            x_traffic_processed = x_traffic

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.traffic_lstm(x_traffic_processed)

        # Process temporal features with proper BatchNorm handling
        temporal_encoded = self.temporal_encoder(x_temporal)

        # Apply BatchNorm if configured, but handle sequence dimension properly
        if self.temporal_bn is not None:
            # Reshape for BatchNorm: (batch * seq, features)
            batch_size, seq_len, features = temporal_encoded.shape
            temporal_encoded_reshaped = temporal_encoded.view(batch_size * seq_len, features)
            temporal_encoded_reshaped = self.temporal_bn(temporal_encoded_reshaped)
            temporal_encoded = temporal_encoded_reshaped.view(batch_size, seq_len, features)

        temporal_context = temporal_encoded.mean(dim=1)

        # Combine bidirectional LSTM outputs
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_n = torch.cat([h_n[-1, 0, :, :], h_n[-1, 1, :, :]], dim=1)

        # Fusion
        combined = torch.cat([h_n, temporal_context], dim=1)
        fused = self.fusion_layer(combined)

        # Attention mechanism
        if self.config.USE_ATTENTION:
            fused_expanded = fused.unsqueeze(1)

            if hasattr(self, 'pos_encoding'):
                lstm_out = self.pos_encoding(lstm_out.transpose(0, 1)).transpose(0, 1)

            attn_out, attn_weights = self.attention(fused_expanded, lstm_out, lstm_out)
            attn_out = attn_out.squeeze(1)
        else:
            attn_out = fused

        # Peak detection - return logits instead of probabilities
        peak_logits = self.peak_detector(attn_out)

        # Generate predictions
        predictions = []
        for i in range(self.pred_horizon):
            pred = self.prediction_heads[i](attn_out)

            # Peak enhancement
            if self.config.USE_RESIDUAL_CONNECTIONS:
                last_value = x_traffic[:, -1, :]
                # Apply sigmoid to logits for peak enhancement
                peak_probs = torch.sigmoid(peak_logits)
                peak_enhancement = self.peak_enhancer(last_value) * peak_probs

                # Combine base prediction with peak enhancement and residual
                pred = pred + self.residual_weight * last_value + \
                       self.peak_residual_weight * peak_enhancement

            predictions.append(pred.unsqueeze(1))

        output = torch.cat(predictions, dim=1)

        return output, peak_logits  # Return logits instead of probabilities


class PeakAwareLoss(nn.Module):
    """Custom loss function that gives more weight to peak periods - FIXED VERSION"""

    def __init__(self, peak_weight=2.0, peak_threshold_percentile=80):
        super(PeakAwareLoss, self).__init__()
        self.peak_weight = peak_weight
        self.peak_threshold_percentile = peak_threshold_percentile
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets, peak_logits=None):
        # Debug prints to understand the shapes
        print(f"DEBUG - Predictions shape: {predictions.shape}")
        print(f"DEBUG - Targets shape: {targets.shape}")

        # Check if shapes match, if not, handle the mismatch
        if predictions.shape != targets.shape:
            print(f"WARNING: Shape mismatch detected!")
            print(f"Predictions: {predictions.shape}, Targets: {targets.shape}")

            # Option 1: Trim predictions to match targets
            if predictions.shape[1] > targets.shape[1]:
                predictions = predictions[:, :targets.shape[1], :]
                print(f"Trimmed predictions to: {predictions.shape}")

            # Option 2: Trim targets to match predictions
            elif targets.shape[1] > predictions.shape[1]:
                targets = targets[:, :predictions.shape[1], :]
                print(f"Trimmed targets to: {targets.shape}")

        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)

        # Identify peak periods
        target_flat = targets.view(-1)
        threshold = torch.quantile(target_flat, self.peak_threshold_percentile / 100.0)
        peak_mask = (targets > threshold).float()

        # Apply peak weighting
        weighted_mse = mse * (1 + peak_mask * (self.peak_weight - 1))
        weighted_mae = mae * (1 + peak_mask * (self.peak_weight - 1))

        # Peak consistency loss
        peak_loss = 0
        if peak_logits is not None:
            batch_avg_targets = targets.mean(dim=(1, 2))
            target_peaks = (batch_avg_targets > threshold).float().unsqueeze(-1)
            peak_loss = F.binary_cross_entropy_with_logits(peak_logits, target_peaks)

        total_loss = weighted_mse.mean() + 0.1 * weighted_mae.mean() + 0.1 * peak_loss
        return total_loss


def train_enhanced_model(model, train_loader, val_loader, config, output_dir):
    """Enhanced training function with better optimization and autocast compatibility"""

    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
    model = model.to(device)

    print(f"Model moved to device: {device}")

    # Enhanced loss function
    if config.USE_PEAK_LOSS:
        criterion = PeakAwareLoss(
            peak_weight=config.PEAK_WEIGHT,
            peak_threshold_percentile=config.PEAK_THRESHOLD_PERCENTILE
        )
    else:
        criterion = nn.MSELoss()

    # Optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Enhanced learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'cosine_warm_restart':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=config.LEARNING_RATE * 0.01
            )
        elif config.SCHEDULER_TYPE == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.LEARNING_RATE,
                epochs=config.NUM_EPOCHS,
                steps_per_epoch=len(train_loader),
                pct_start=0.3
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )

    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Mixed precision training - disable for compatibility
    use_mixed_precision = False
    scaler = None

    # Check PyTorch version for mixed precision compatibility
    pytorch_version = torch.__version__
    if pytorch_version >= "2.0.0":
        try:
            if hasattr(TrainingConfig, 'MIXED_PRECISION') and TrainingConfig.MIXED_PRECISION:
                scaler = torch.amp.GradScaler('cuda')
                use_mixed_precision = True
                print("Using modern mixed precision training")
        except:
            print("Mixed precision not available, using standard training")
    else:
        print("PyTorch < 2.0, disabling mixed precision for compatibility")

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')

        for batch_idx, (batch_x_traffic, batch_x_temporal, batch_y) in enumerate(progress_bar):
            batch_x_traffic = batch_x_traffic.to(device)
            batch_x_temporal = batch_x_temporal.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass - simplified without problematic mixed precision
            if config.USE_PEAK_LOSS:
                output, peak_logits = model(batch_x_traffic, batch_x_temporal)
                loss = criterion(output, batch_y, peak_logits)
            else:
                output, _ = model(batch_x_traffic, batch_x_temporal)
                loss = criterion(output, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           getattr(TrainingConfig, 'GRADIENT_CLIP_VALUE', 1.0))
            optimizer.step()

            if scheduler and config.SCHEDULER_TYPE == 'onecycle':
                scheduler.step()

            train_loss += loss.item()
            train_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})

            # Log frequently during training
            if batch_idx % config.LOG_FREQUENCY == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch_x_traffic, batch_x_temporal, batch_y in val_loader:
                batch_x_traffic = batch_x_traffic.to(device)
                batch_x_temporal = batch_x_temporal.to(device)
                batch_y = batch_y.to(device)

                if config.USE_PEAK_LOSS:
                    output, peak_logits = model(batch_x_traffic, batch_x_temporal)
                    loss = criterion(output, batch_y, peak_logits)
                else:
                    output, _ = model(batch_x_traffic, batch_x_temporal)
                    loss = criterion(output, batch_y)

                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        if scheduler and config.SCHEDULER_TYPE not in ['onecycle']:
            if config.SCHEDULER_TYPE == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        logger.info(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')

        # Save checkpoints
        if config.SAVE_CHECKPOINTS and (epoch + 1) % config.CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0

            if config.SAVE_BEST_MODEL:
                model_path = os.path.join(output_dir, 'models', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config
                }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # Load best model
    if config.SAVE_BEST_MODEL:
        model_path = os.path.join(output_dir, 'models', 'best_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


def main():
    """Main training function with enhanced model"""

    # Create output directory and save config
    output_dir = Config.create_output_folder()
    config_path = Config.save_config(output_dir)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration saved to: {config_path}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and Config.USE_CUDA else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading preprocessed data...")
    with open(Config.DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Extract components
    X_traffic_train = data['X_traffic_train']
    X_temporal_train = data['X_temporal_train']
    y_train = data['y_train']

    X_traffic_val = data['X_traffic_val']
    X_temporal_val = data['X_temporal_val']
    y_val = data['y_val']

    X_traffic_test = data['X_traffic_test']
    X_temporal_test = data['X_temporal_test']
    y_test = data['y_test']

    scaler = data['scaler']

    logger.info(f"Dataset shapes:")
    logger.info(f"  Train: {X_traffic_train.shape}")
    logger.info(f"  Val: {X_traffic_val.shape}")
    logger.info(f"  Test: {X_traffic_test.shape}")

    # Create enhanced datasets with augmentation
    train_dataset = MilanTrafficDataset(
        X_traffic_train, X_temporal_train, y_train,
        augment=True, config=Config
    )
    val_dataset = MilanTrafficDataset(X_traffic_val, X_temporal_val, y_val)
    test_dataset = MilanTrafficDataset(X_traffic_test, X_temporal_test, y_test)

    # Enhanced data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    # Initialize enhanced model
    model = EnhancedLSTMTrafficPredictor(
        n_squares=X_traffic_train.shape[2],
        n_temporal_features=X_temporal_train.shape[2],
        config=Config
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Enhanced LSTM Model initialized")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Train the enhanced model
    logger.info("Starting enhanced training...")
    train_losses, val_losses = train_enhanced_model(
        model, train_loader, val_loader, Config, output_dir
    )

    # Save training history
    history_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': Config.__dict__
    }

    history_path = os.path.join(output_dir, 'metrics', 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4, default=str)

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training History - Log Scale')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_history.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Simple metrics calculation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"Best Validation Loss: {min(val_losses):.6f}")
    print(f"Best Epoch: {np.argmin(val_losses) + 1}")
    print(f"Total Epochs: {len(train_losses)}")
    print(f"Model Parameters: {total_params:,}")
    print(f"Results saved to: {output_dir}")

    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'final_model.pth')
    final_model_path = save_model_safely(
        model, train_losses, val_losses,
        total_params, trainable_params,
        output_dir, Config
    )

    logger.info(f"Enhanced model training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")

    return train_losses, val_losses


if __name__ == "__main__":
    main()