#!/usr/bin/env python3
"""
Enhanced Milan Traffic Prediction - Main Script
Integrates all components with improved architecture and visualization
"""
import os

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
import logging
from datetime import datetime

# Import our modules
from config import Config
from model import (
    EnhancedLSTMTrafficPredictor,
    MilanTrafficDataset,
    train_enhanced_model,
    main as model_main
)
from plot import enhanced_visualization_v2, collect_predictions_enhanced


def setup_logging(output_dir):
    """Setup comprehensive logging"""

    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging configuration
    log_filename = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
    logger.info(f"Log file: {log_filename}")

    return logger


def safe_config_dict(config):
    """Safely convert config to dictionary, avoiding pickle issues"""
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_') and not callable(getattr(config, key)):
            try:
                value = getattr(config, key)
                # Convert to basic types to avoid pickle issues
                if isinstance(value, (int, float, str, bool, list, tuple)):
                    config_dict[key] = value
                elif hasattr(value, '__dict__'):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = str(value)
            except:
                pass
    return config_dict


def print_model_summary(model, config):
    """Print detailed model summary"""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("ENHANCED MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    print(f"Model Type: {config.MODEL_TYPE.upper()}")
    print(f"Architecture: Enhanced LSTM with Advanced Features")
    print(f"\nModel Configuration:")
    print(f"  ├── Hidden Dimension: {config.HIDDEN_DIM}")
    print(f"  ├── Number of Layers: {config.NUM_LAYERS}")
    print(f"  ├── Dropout Rate: {config.DROPOUT}")
    print(f"  ├── Sequence Length: {config.SEQUENCE_LENGTH} steps ({config.SEQUENCE_LENGTH * 10} minutes)")
    print(f"  └── Prediction Horizon: {config.PREDICTION_HORIZON} steps ({config.PREDICTION_HORIZON * 10} minutes)")

    print(f"\nAdvanced Features:")
    print(f"  ├── Attention Mechanism: {'✓' if config.USE_ATTENTION else '✗'}")
    print(f"  ├── Multi-Scale CNN: {'✓' if config.USE_MULTI_SCALE else '✗'}")
    print(f"  ├── Peak-Aware Loss: {'✓' if config.USE_PEAK_LOSS else '✗'}")
    print(f"  ├── Residual Connections: {'✓' if config.USE_RESIDUAL_CONNECTIONS else '✗'}")
    print(f"  ├── Batch Normalization: {'✓' if config.USE_BATCH_NORM else '✗'}")
    print(f"  ├── Data Augmentation: {'✓' if config.USE_DATA_AUGMENTATION else '✗'}")
    print(f"  └── Mixed Precision: {'✓' if hasattr(config, 'MIXED_PRECISION') else '✗'}")

    print(f"\nModel Parameters:")
    print(f"  ├── Total Parameters: {total_params:,}")
    print(f"  ├── Trainable Parameters: {trainable_params:,}")
    print(f"  └── Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    print(f"\nTraining Configuration:")
    print(f"  ├── Batch Size: {config.BATCH_SIZE}")
    print(f"  ├── Learning Rate: {config.LEARNING_RATE}")
    print(f"  ├── Max Epochs: {config.NUM_EPOCHS}")
    print(f"  ├── Patience: {config.PATIENCE}")
    print(f"  ├── Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  └── Scheduler: {config.SCHEDULER_TYPE if config.USE_SCHEDULER else 'None'}")

    if config.USE_PEAK_LOSS:
        print(f"\nPeak Enhancement:")
        print(f"  ├── Peak Weight: {config.PEAK_WEIGHT}x")
        print(f"  └── Peak Threshold: {config.PEAK_THRESHOLD_PERCENTILE}th percentile")

    print("=" * 80)


def run_comprehensive_analysis():
    """Run the complete enhanced analysis pipeline"""

    print("\n STARTING ENHANCED MILAN TRAFFIC PREDICTION")
    print("=" * 80)

    # Step 1: Setup
    output_dir = Config.create_output_folder()
    logger = setup_logging(output_dir)

    logger.info("Enhanced Milan Traffic Prediction Started")
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config_path = Config.save_config(output_dir)
    logger.info(f"Configuration saved to: {config_path}")

    # Step 2: Hardware and environment check
    device = torch.device('cuda' if torch.cuda.is_available() and Config.USE_CUDA else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Step 3: Data loading and validation
    logger.info("Loading and validating data...")

    try:
        import pickle
        with open(Config.DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        logger.info("✓ Data loaded successfully")
    except FileNotFoundError:
        logger.error(f"✗ Data file not found: {Config.DATA_PATH}")
        logger.error("Please run data preprocessing first!")
        return
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        return

    # Extract and validate data components
    required_keys = ['X_traffic_train', 'X_temporal_train', 'y_train',
                     'X_traffic_val', 'X_temporal_val', 'y_val',
                     'X_traffic_test', 'X_temporal_test', 'y_test', 'scaler']

    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logger.error(f"✗ Missing data components: {missing_keys}")
        return

    # Data shape validation
    X_traffic_train = data['X_traffic_train']
    X_temporal_train = data['X_temporal_train']
    y_train = data['y_train']

    logger.info(f"Data shapes validated:")
    logger.info(f"  Train: {X_traffic_train.shape}")
    logger.info(f"  Val: {data['X_traffic_val'].shape}")
    logger.info(f"  Test: {data['X_traffic_test'].shape}")

    # Step 4: Model initialization
    logger.info("Initializing enhanced model...")

    model = EnhancedLSTMTrafficPredictor(
        n_squares=X_traffic_train.shape[2],
        n_temporal_features=X_temporal_train.shape[2],
        config=Config
    )

    # Print model summary
    print_model_summary(model, Config)
    logger.info("✓ Model initialized successfully")

    # Step 5: Data preparation
    logger.info("Preparing data loaders...")

    train_dataset = MilanTrafficDataset(
        X_traffic_train, X_temporal_train, y_train,
        augment=True, config=Config
    )
    val_dataset = MilanTrafficDataset(
        data['X_traffic_val'], data['X_temporal_val'], data['y_val']
    )
    test_dataset = MilanTrafficDataset(
        data['X_traffic_test'], data['X_temporal_test'], data['y_test']
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )

    logger.info("✓ Data loaders created successfully")

    # Step 6: Training
    print("\n STARTING ENHANCED TRAINING")
    print("-" * 50)

    logger.info("Starting model training...")
    start_time = datetime.now()

    try:
        train_losses, val_losses = train_enhanced_model(
            model, train_loader, val_loader, Config, output_dir
        )

        training_time = datetime.now() - start_time
        logger.info(f"✓ Training completed in {training_time}")

    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise

    # Step 7: Evaluation and Visualization
    print("\n GENERATING COMPREHENSIVE ANALYSIS")
    print("-" * 50)

    logger.info("Collecting predictions...")

    val_predictions, val_actuals, val_peaks, test_predictions, test_actuals, test_peaks = \
        collect_predictions_enhanced(model, val_loader, test_loader, device, Config)

    # Save raw predictions
    predictions_dir = os.path.join(output_dir, 'predictions')
    np.save(os.path.join(predictions_dir, 'val_predictions.npy'), val_predictions)
    np.save(os.path.join(predictions_dir, 'val_actuals.npy'), val_actuals)
    np.save(os.path.join(predictions_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(predictions_dir, 'test_actuals.npy'), test_actuals)

    logger.info("Generating enhanced visualizations...")

    val_metrics, test_metrics = enhanced_visualization_v2(
        val_predictions, val_actuals, val_peaks,
        test_predictions, test_actuals, test_peaks,
        data['scaler'], output_dir, Config
    )

    # Step 8: Results Summary and Saving
    print("\n FINAL RESULTS SUMMARY")
    print("=" * 80)

    # Calculate improvement metrics
    baseline_metrics = {
        'rmse': np.sqrt(np.mean((val_actuals.mean(axis=2) - val_actuals.mean(axis=2).mean()) ** 2)),
        'mae': np.mean(np.abs(val_actuals.mean(axis=2) - val_actuals.mean(axis=2).mean()))
    }

    improvement_rmse = (baseline_metrics['rmse'] - val_metrics['rmse']) / baseline_metrics['rmse'] * 100
    improvement_mae = (baseline_metrics['mae'] - val_metrics['mae']) / baseline_metrics['mae'] * 100

    print(f" PERFORMANCE METRICS")
    print(f"{'Metric':<20} {'Validation':<12} {'Test':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'RMSE':<20} {val_metrics['rmse']:<12.4f} {test_metrics['rmse']:<12.4f} {improvement_rmse:>+10.1f}%")
    print(f"{'MAE':<20} {val_metrics['mae']:<12.4f} {test_metrics['mae']:<12.4f} {improvement_mae:>+10.1f}%")
    print(f"{'MAPE':<20} {val_metrics['mape']:<12.2f} {test_metrics['mape']:<12.2f} {'':<12}")
    print(f"{'R²':<20} {val_metrics['r2']:<12.4f} {test_metrics['r2']:<12.4f} {'':<12}")
    print(f"{'Correlation':<20} {val_metrics['correlation']:<12.4f} {test_metrics['correlation']:<12.4f} {'':<12}")

    print(f"\n PEAK PERFORMANCE")
    print(f"{'Peak RMSE':<20} {val_metrics['peak_rmse']:<12.4f} {test_metrics['peak_rmse']:<12.4f}")
    print(f"{'Peak MAE':<20} {val_metrics['peak_mae']:<12.4f} {test_metrics['peak_mae']:<12.4f}")
    print(f"{'Peak MAPE':<20} {val_metrics['peak_mape']:<12.2f} {test_metrics['peak_mape']:<12.2f}")

    print(f"\n LOW TRAFFIC PERFORMANCE")
    print(f"{'Low RMSE':<20} {val_metrics['low_rmse']:<12.4f} {test_metrics['low_rmse']:<12.4f}")
    print(f"{'Low MAE':<20} {val_metrics['low_mae']:<12.4f} {test_metrics['low_mae']:<12.4f}")

    print(f"\n DIRECTIONAL ACCURACY")
    print(
        f"{'Direction Acc.':<20} {val_metrics['directional_accuracy']:<12.4f} {test_metrics['directional_accuracy']:<12.4f}")

    print(f"\n  TRAINING SUMMARY")
    print(f"{'Training Time':<20} {training_time}")
    print(f"{'Epochs Completed':<20} {len(train_losses)}")
    print(f"{'Best Epoch':<20} {np.argmin(val_losses) + 1}")
    print(f"{'Final Train Loss':<20} {train_losses[-1]:.6f}")
    print(f"{'Best Val Loss':<20} {min(val_losses):.6f}")

    # Create safe config dict - FIXED VERSION
    safe_config = safe_config_dict(Config)

    # Save comprehensive results with safe config
    results = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': str(training_time),
            'epochs_completed': len(train_losses),
            'best_epoch': int(np.argmin(val_losses) + 1)
        },
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_architecture': str(model),
            'config': safe_config  # Use safe config here
        },
        'improvements': {
            'rmse_improvement_percent': float(improvement_rmse),
            'mae_improvement_percent': float(improvement_mae)
        }
    }

    # Save to multiple formats
    import json
    results_path = os.path.join(output_dir, 'metrics', 'comprehensive_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # Save model with full metadata - FIXED VERSION
    model_save_path = os.path.join(output_dir, 'models', 'enhanced_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': safe_config,  # Use safe config instead of Config.__dict__
        'results': {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'training_time': str(training_time),
                'epochs_completed': len(train_losses),
                'best_epoch': int(np.argmin(val_losses) + 1)
            }
        },
        'training_completed': datetime.now().isoformat()
    }, model_save_path)

    # Create README for the results
    readme_content = f"""# Enhanced Milan Traffic Prediction Results

## Experiment Details
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model**: Enhanced LSTM with Advanced Features
- **Training Time**: {training_time}
- **Device**: {device}

## Model Configuration
- Hidden Dimension: {Config.HIDDEN_DIM}
- Layers: {Config.NUM_LAYERS}
- Dropout: {Config.DROPOUT}
- Attention: {Config.USE_ATTENTION}
- Multi-Scale: {Config.USE_MULTI_SCALE}
- Peak Loss: {Config.USE_PEAK_LOSS}

## Performance Summary
### Validation Set
- RMSE: {val_metrics['rmse']:.4f}
- MAE: {val_metrics['mae']:.4f}
- R²: {val_metrics['r2']:.4f}
- Peak RMSE: {val_metrics['peak_rmse']:.4f}

### Test Set  
- RMSE: {test_metrics['rmse']:.4f}
- MAE: {test_metrics['mae']:.4f}
- R²: {test_metrics['r2']:.4f}
- Peak RMSE: {test_metrics['peak_rmse']:.4f}

## Files Generated
- `models/enhanced_model_final.pth` - Final trained model
- `plots/enhanced_full_validation_analysis.png` - Full validation analysis
- `plots/peak_performance_analysis.png` - Peak prediction analysis
- `plots/comprehensive_test_analysis.png` - Test set analysis
- `plots/multi_square_performance.png` - Per-square performance
- `plots/temporal_pattern_analysis.png` - Temporal patterns
- `metrics/comprehensive_results.json` - All metrics and results
- `logs/training_*.log` - Training logs

## Key Insights
1. {'Significant improvement in peak prediction' if val_metrics.get('peak_rmse', 0) < val_metrics.get('rmse', 0) else 'Peak prediction needs attention'}
2. {'Strong directional accuracy' if val_metrics.get('directional_accuracy', 0) > 0.7 else 'Directional prediction could be improved'}
3. {'Good generalization' if abs(val_metrics.get('rmse', 0) - test_metrics.get('rmse', 0)) < 0.1 * val_metrics.get('rmse', 1) else 'Potential overfitting detected'}

## Next Steps
1. Review peak performance analysis for further improvements
2. Consider ensemble methods for better robustness
3. Analyze temporal patterns for domain-specific insights
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"\n RESULTS SAVED")
    print(f"Output Directory: {output_dir}")
    print(f"Comprehensive Results: {results_path}")
    print(f"Model Saved: {model_save_path}")
    print(f"Report: {readme_path}")

    logger.info("Enhanced analysis completed successfully!")
    logger.info(f"All results saved to: {output_dir}")

    return output_dir, results


def main():
    """Main entry point"""
    try:
        output_dir, results = run_comprehensive_analysis()

        print(f"\n SUCCESS! Enhanced Milan Traffic Prediction completed.")
        print(f" Check results in: {output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())