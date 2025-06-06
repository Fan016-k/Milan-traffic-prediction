"""
Configuration file for Milan Traffic Prediction - MAXIMUM PERFORMANCE OPTIMIZATION
Advanced hyperparameters tuned for state-of-the-art results
"""

import os
from datetime import datetime

class Config:
    # Data Configuration
    DATA_PATH = r'C:\Users\Fan\PycharmProjects\dataset2\processed_data\milan_traffic_data_processed_full_memory_efficient.pkl'
    DATA_DIRECTORY = r"C:\Users\Fan\PycharmProjects\dataset2\dataverse_files"

    # Model Architecture Configuration
    MODEL_TYPE = 'enhanced_lstm'

    # MAXIMUM CAPACITY LSTM PARAMETERS - HEAVY ARCHITECTURE
    HIDDEN_DIM = 512         # MAJOR increase from 384 (significantly more capacity)
    NUM_LAYERS = 5           # Deeper network from 4 (more complex patterns)
    DROPOUT = 0.15           # REDUCED from 0.25 (let the model learn more)

    # ADVANCED FEATURES - ALL ENABLED
    USE_ATTENTION = True
    USE_RESIDUAL_CONNECTIONS = True
    USE_BATCH_NORM = True
    USE_LAYER_NORM = True

    # MULTI-SCALE PROCESSING - MAXIMUM COVERAGE
    USE_MULTI_SCALE = True
    KERNEL_SIZES = [3, 5, 7, 11, 15]  # Added even larger kernel (15) for very long patterns

    # PEAK ENHANCEMENT - AGGRESSIVE FOCUS
    USE_PEAK_LOSS = True
    PEAK_WEIGHT = 5.0         # MAJOR increase from 3.0 (much stronger peak focus)
    PEAK_THRESHOLD_PERCENTILE = 70  # Even lower threshold = capture more peaks

    # TRAINING CONFIGURATION - ULTRA OPTIMIZED
    BATCH_SIZE = 16          # Even smaller for better gradient quality
    NUM_EPOCHS = 300         # Much more epochs for full convergence
    LEARNING_RATE = 0.0001   # LOWER for ultra-stable training
    WEIGHT_DECAY = 1e-5      # Minimal regularization for max learning

    # ADVANCED LEARNING RATE SCHEDULING - OPTIMAL STRATEGY
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine_warm_restart'  # Better than onecycle for long training
    WARMUP_EPOCHS = 25       # Much longer warmup for stability

    # EARLY STOPPING - ULTRA PATIENT
    PATIENCE = 50            # Maximum patience for complex convergence
    MIN_DELTA = 1e-7         # Ultra-fine improvements detection

    # DATA PREPROCESSING - MAXIMUM CONTEXT
    SEQUENCE_LENGTH = 240    # 40 hours from 28 (capture weekly patterns)
    PREDICTION_HORIZON = 18  # 3 hours from 2 (more challenging and useful)
    SCALER_TYPE = 'robust'   # Robust to outliers

    # DATA AUGMENTATION - AGGRESSIVE REGULARIZATION
    USE_DATA_AUGMENTATION = True
    NOISE_FACTOR = 0.025     # More noise for stronger robustness
    TIME_SHIFT_RANGE = 12    # Stronger temporal augmentation

    # OUTPUT CONFIGURATION
    OUTPUT_BASE_PATH = '../output'
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_FREQUENCY = 20  # Save even less frequently

    # VISUALIZATION CONFIGURATION - HIGH QUALITY
    VIZ_SAMPLE_SIZE = 5000    # Even more samples for detailed analysis
    VIZ_ZOOM_STEPS = 2000     # Longer zoom views
    VIZ_WEEKLY_ANALYSIS = True
    VIZ_SAVE_DPI = 400        # Higher DPI for publication quality

    # HARDWARE CONFIGURATION
    USE_CUDA = True
    NUM_WORKERS = 1          # Single worker for maximum stability
    PIN_MEMORY = True

    # LOGGING
    LOG_LEVEL = 'INFO'
    LOG_FREQUENCY = 20       # Even more frequent logging

    # EVALUATION METRICS
    METRICS = ['mse', 'rmse', 'mae', 'mape', 'r2', 'nrmse', 'smape', 'directional_accuracy']

    @classmethod
    def create_output_folder(cls):
        """Create output folder with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(cls.OUTPUT_BASE_PATH, f'milan_traffic_ULTRA_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        for subdir in ['models', 'plots', 'metrics', 'predictions', 'checkpoints', 'logs']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        return output_dir

    @classmethod
    def save_config(cls, output_dir):
        """Save configuration to file"""
        import json

        config_dict = {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }

        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

        return config_path

# MODEL-SPECIFIC CONFIGURATIONS - MAXIMUM CAPACITY
class LSTMConfig:
    """LSTM-specific configuration - HEAVY ARCHITECTURE"""
    BIDIRECTIONAL = True
    HIDDEN_DIM = 512         # Match main config
    NUM_LAYERS = 5           # Match main config
    DROPOUT = 0.15           # Match main config

class AttentionConfig:
    """Attention mechanism configuration - MAXIMUM POWER"""
    NUM_HEADS = 16           # MAJOR increase from 12 (more attention complexity)
    ATTENTION_DIM = 512      # Match hidden dimension
    DROPOUT = 0.05           # Less dropout in attention
    USE_POSITIONAL_ENCODING = True
    ATTENTION_LAYERS = 2     # Multiple attention layers

class TrainingConfig:
    """Training-specific configuration - ULTRA OPTIMIZED"""
    GRADIENT_CLIP_VALUE = 0.3    # Even gentler clipping
    ACCUMULATION_STEPS = 4       # More gradient accumulation for larger effective batch
    MIXED_PRECISION = False      # Keep disabled for stability

    # LOSS FUNCTION WEIGHTS - PEAK-FOCUSED
    MSE_WEIGHT = 1.0
    MAE_WEIGHT = 0.2         # Even less MAE influence
    PEAK_LOSS_WEIGHT = 5.0   # Match peak weight
    DIRECTIONAL_LOSS_WEIGHT = 0.1  # Add directional accuracy loss

    # ADVANCED TRAINING TECHNIQUES
    LABEL_SMOOTHING = 0.05   # Small label smoothing for robustness
    CUTMIX_ALPHA = 0.2       # Temporal cutmix augmentation
    MIXUP_ALPHA = 0.3        # Temporal mixup augmentation

# ADVANCED EXPERIMENTAL FEATURES - ALL ENABLED
class ExperimentalConfig:
    """Experimental features for breakthrough performance"""

    # ADVANCED FEATURE ENGINEERING
    USE_FOURIER_FEATURES = True    # Frequency domain features
    USE_CYCLICAL_ENCODING = True   # Better time encoding
    USE_WAVELET_FEATURES = True    # Wavelet decomposition features
    USE_LAG_FEATURES = True        # Multiple lag features

    # ATTENTION ENHANCEMENTS
    USE_SELF_ATTENTION = True      # Self-attention within sequences
    USE_CROSS_ATTENTION = True     # Cross-attention between traffic/temporal
    USE_MULTI_HEAD_ATTENTION = True

    # ADVANCED REGULARIZATION
    USE_DROPOUT_SCHEDULE = True    # Adaptive dropout
    USE_STOCHASTIC_DEPTH = True    # Stochastic depth for deeper training
    USE_SPECTRAL_NORM = True       # Spectral normalization

    # ENSEMBLE TECHNIQUES
    USE_TEACHER_FORCING = True     # Curriculum learning
    USE_SELF_DISTILLATION = True   # Knowledge distillation
    USE_ENSEMBLE_PREDICTION = True # Multiple prediction heads

    # LOSS FUNCTION ENHANCEMENTS
    USE_FOCAL_LOSS = True          # Focal loss for hard examples
    USE_QUANTILE_LOSS = True       # Quantile regression
    USE_HUBER_LOSS = True          # Huber loss for robustness

    # ADVANCED OPTIMIZATIONS
    USE_LOOKAHEAD_OPTIMIZER = True # Lookahead optimizer wrapper
    USE_SAM_OPTIMIZER = True       # Sharpness-Aware Minimization
    USE_ADABELIEF = True           # AdaBelief optimizer

# PEAK PERFORMANCE SPECIALIZED CONFIG
class PeakConfig:
    """Specialized configuration for maximum peak prediction accuracy"""

    # Peak detection thresholds
    PEAK_PERCENTILES = [70, 80, 90, 95]  # Multiple peak thresholds
    LOW_TRAFFIC_PERCENTILE = 20          # Low traffic threshold

    # Peak-specific losses
    PEAK_FOCAL_GAMMA = 2.0               # Focal loss gamma for peaks
    PEAK_WEIGHTED_LOSS = True            # Weighted loss by traffic level

    # Peak enhancement
    PEAK_ATTENTION_HEADS = 8             # Dedicated attention for peaks
    PEAK_PREDICTION_LAYERS = 3           # Separate peak prediction pathway

    # Dynamic weighting
    USE_DYNAMIC_PEAK_WEIGHT = True       # Adjust peak weight during training
    PEAK_WEIGHT_SCHEDULE = 'cosine'      # Peak weight scheduling

# ADVANCED METRICS CONFIG
class MetricsConfig:
    """Advanced metrics for comprehensive evaluation"""

    # Standard metrics
    BASIC_METRICS = ['mse', 'rmse', 'mae', 'mape', 'r2', 'nrmse']

    # Peak-specific metrics
    PEAK_METRICS = ['peak_rmse', 'peak_mae', 'peak_mape', 'peak_recall', 'peak_precision']

    # Directional metrics
    DIRECTIONAL_METRICS = ['directional_accuracy', 'trend_accuracy', 'turning_point_detection']

    # Distribution metrics
    DISTRIBUTION_METRICS = ['ks_test', 'wasserstein_distance', 'energy_distance']

    # Temporal metrics
    TEMPORAL_METRICS = ['temporal_correlation', 'phase_coherence', 'spectral_similarity']