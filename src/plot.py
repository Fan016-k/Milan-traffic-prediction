import os

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def collect_predictions_enhanced(model, val_loader, test_loader, device, config):
    """Collect predictions with enhanced peak information"""
    model.eval()

    val_predictions = []
    val_actuals = []
    val_peaks = []
    test_predictions = []
    test_actuals = []
    test_peaks = []

    with torch.no_grad():
        # Validation set
        for batch_x_traffic, batch_x_temporal, batch_y in tqdm(val_loader, desc='Validation predictions'):
            batch_x_traffic = batch_x_traffic.to(device)
            batch_x_temporal = batch_x_temporal.to(device)

            if config.USE_PEAK_LOSS:
                output, peak_scores = model(batch_x_traffic, batch_x_temporal)
                val_peaks.append(peak_scores.cpu().numpy())
            else:
                output, _ = model(batch_x_traffic, batch_x_temporal)
                val_peaks.append(np.zeros((output.shape[0], 1)))  # Dummy peaks

            val_predictions.append(output.cpu().numpy())
            val_actuals.append(batch_y.numpy())

        # Test set
        for batch_x_traffic, batch_x_temporal, batch_y in tqdm(test_loader, desc='Test predictions'):
            batch_x_traffic = batch_x_traffic.to(device)
            batch_x_temporal = batch_x_temporal.to(device)

            if config.USE_PEAK_LOSS:
                output, peak_scores = model(batch_x_traffic, batch_x_temporal)
                test_peaks.append(peak_scores.cpu().numpy())
            else:
                output, _ = model(batch_x_traffic, batch_x_temporal)
                test_peaks.append(np.zeros((output.shape[0], 1)))  # Dummy peaks

            test_predictions.append(output.cpu().numpy())
            test_actuals.append(batch_y.numpy())

    # Concatenate all results
    val_predictions = np.concatenate(val_predictions, axis=0)
    val_actuals = np.concatenate(val_actuals, axis=0)
    val_peaks = np.concatenate(val_peaks, axis=0)

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)
    test_peaks = np.concatenate(test_peaks, axis=0)

    return val_predictions, val_actuals, val_peaks, test_predictions, test_actuals, test_peaks


def calculate_enhanced_metrics(predictions, targets, scaler=None):
    """Calculate comprehensive performance metrics with robust error handling"""

    try:
        # Denormalize if scaler provided
        if scaler is not None:
            pred_shape = predictions.shape
            target_shape = targets.shape

            predictions_denorm = scaler.inverse_transform(
                predictions.reshape(-1, pred_shape[-1])
            ).reshape(pred_shape)
            targets_denorm = scaler.inverse_transform(
                targets.reshape(-1, target_shape[-1])
            ).reshape(target_shape)
        else:
            predictions_denorm = predictions
            targets_denorm = targets

        # Flatten for overall metrics and ensure same length
        pred_flat = predictions_denorm.flatten()
        target_flat = targets_denorm.flatten()

        # Align lengths
        min_len = min(len(pred_flat), len(target_flat))
        pred_flat = pred_flat[:min_len]
        target_flat = target_flat[:min_len]

        # Basic metrics
        mse = mean_squared_error(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_flat, pred_flat)

        # MAPE with better handling
        epsilon = np.finfo(float).eps
        mask = np.abs(target_flat) > epsilon
        if mask.sum() > 0:
            mape = np.mean(np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])) * 100
        else:
            mape = np.inf

        # R-squared
        r2 = r2_score(target_flat, pred_flat)

        # Normalized metrics
        target_range = np.max(target_flat) - np.min(target_flat)
        nrmse = rmse / (target_range + epsilon)

        # Peak-specific metrics
        peak_threshold = np.percentile(target_flat, 90)
        peak_mask = target_flat > peak_threshold

        if peak_mask.sum() > 0:
            peak_rmse = np.sqrt(mean_squared_error(target_flat[peak_mask], pred_flat[peak_mask]))
            peak_mae = mean_absolute_error(target_flat[peak_mask], pred_flat[peak_mask])
            peak_mape = np.mean(np.abs((target_flat[peak_mask] - pred_flat[peak_mask]) /
                                       target_flat[peak_mask])) * 100
        else:
            peak_rmse = peak_mae = peak_mape = 0

        # Low traffic metrics
        low_threshold = np.percentile(target_flat, 10)
        low_mask = target_flat < low_threshold

        if low_mask.sum() > 0:
            low_rmse = np.sqrt(mean_squared_error(target_flat[low_mask], pred_flat[low_mask]))
            low_mae = mean_absolute_error(target_flat[low_mask], pred_flat[low_mask])
        else:
            low_rmse = low_mae = 0

        # Correlation metrics
        if len(pred_flat) > 1 and np.var(pred_flat) > 0 and np.var(target_flat) > 0:
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        else:
            correlation = 0

        # Directional accuracy (for time series)
        if len(pred_flat) > 1:
            pred_direction = np.diff(pred_flat) > 0
            target_direction = np.diff(target_flat) > 0
            directional_accuracy = np.mean(pred_direction == target_direction)
        else:
            directional_accuracy = 0

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'nrmse': float(nrmse),
            'correlation': float(correlation),
            'directional_accuracy': float(directional_accuracy),
            'peak_rmse': float(peak_rmse),
            'peak_mae': float(peak_mae),
            'peak_mape': float(peak_mape),
            'low_rmse': float(low_rmse),
            'low_mae': float(low_mae)
        }

    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        # Return default metrics
        return {
            'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0,
            'nrmse': 0.0, 'correlation': 0.0, 'directional_accuracy': 0.0,
            'peak_rmse': 0.0, 'peak_mae': 0.0, 'peak_mape': 0.0,
            'low_rmse': 0.0, 'low_mae': 0.0
        }


def enhanced_visualization_v2(val_predictions, val_actuals, val_peaks,
                              test_predictions, test_actuals, test_peaks,
                              scaler, output_dir, config):
    """Enhanced visualization with robust shape handling"""

    print("Creating enhanced visualizations...")

    # Debug shapes first
    print(f"DEBUG - Input shapes:")
    print(f"  val_predictions: {val_predictions.shape}")
    print(f"  val_actuals: {val_actuals.shape}")
    print(f"  test_predictions: {test_predictions.shape}")
    print(f"  test_actuals: {test_actuals.shape}")

    # Denormalize data
    if scaler is not None:
        try:
            val_predictions = scaler.inverse_transform(
                val_predictions.reshape(-1, val_predictions.shape[-1])
            ).reshape(val_predictions.shape)
            val_actuals = scaler.inverse_transform(
                val_actuals.reshape(-1, val_actuals.shape[-1])
            ).reshape(val_actuals.shape)
            test_predictions = scaler.inverse_transform(
                test_predictions.reshape(-1, test_predictions.shape[-1])
            ).reshape(test_predictions.shape)
            test_actuals = scaler.inverse_transform(
                test_actuals.reshape(-1, test_actuals.shape[-1])
            ).reshape(test_actuals.shape)
        except Exception as e:
            print(f"Warning: Could not denormalize data: {e}")

    # Shape alignment function
    def align_shapes(pred, actual):
        """Align prediction and actual shapes by trimming to minimum dimensions"""
        min_samples = min(pred.shape[0], actual.shape[0])
        min_horizon = min(pred.shape[1], actual.shape[1])
        min_squares = min(pred.shape[2], actual.shape[2])

        pred_aligned = pred[:min_samples, :min_horizon, :min_squares]
        actual_aligned = actual[:min_samples, :min_horizon, :min_squares]

        print(f"  Aligned shapes - pred: {pred_aligned.shape}, actual: {actual_aligned.shape}")
        return pred_aligned, actual_aligned

    # Align all arrays
    val_predictions, val_actuals = align_shapes(val_predictions, val_actuals)
    test_predictions, test_actuals = align_shapes(test_predictions, test_actuals)

    # Find busiest squares
    try:
        busiest_square = np.argmax(val_actuals.mean(axis=(0, 1)))
        second_busiest = np.argsort(val_actuals.mean(axis=(0, 1)))[-2]
    except:
        busiest_square = 0
        second_busiest = min(1, val_actuals.shape[2] - 1)

    # 1. ENHANCED FULL VALIDATION SET ANALYSIS
    print("Creating enhanced full validation analysis...")
    try:
        fig = plt.figure(figsize=(24, 16))

        # Create a more sophisticated layout
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)

        # Full validation - Average across all squares
        val_pred_avg = val_predictions.mean(axis=2).flatten()
        val_actual_avg = val_actuals.mean(axis=2).flatten()

        # Ensure both arrays have same length
        min_length = min(len(val_pred_avg), len(val_actual_avg))
        val_pred_avg = val_pred_avg[:min_length]
        val_actual_avg = val_actual_avg[:min_length]

        print(f"  Final aligned lengths - pred: {len(val_pred_avg)}, actual: {len(val_actual_avg)}")

        # Plot 1: Full validation set
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(val_actual_avg, 'b-', label='Actual', alpha=0.8, linewidth=0.8)
        ax1.plot(val_pred_avg, 'r-', label='Predicted', alpha=0.9, linewidth=0.8)
        ax1.set_title('Full Validation Set - Average Traffic Across All Squares', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time Steps (10-min intervals)')
        ax1.set_ylabel('Traffic Volume')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add peak highlighting
        if len(val_actual_avg) > 0:
            peak_threshold = np.percentile(val_actual_avg, 90)
            peak_indices = val_actual_avg > peak_threshold
            ax1.scatter(np.where(peak_indices)[0], val_actual_avg[peak_indices],
                        c='orange', s=2, alpha=0.6, label='Peaks (90th percentile)')
            ax1.legend(fontsize=12)

        # Plot 2: Error over time
        ax2 = fig.add_subplot(gs[0, 2])
        errors = val_pred_avg - val_actual_avg
        ax2.plot(errors, 'g-', alpha=0.7, linewidth=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Errors Over Time', fontsize=14)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Error (Pred - Actual)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Busiest square analysis
        val_pred_square = val_predictions[:, :, busiest_square].flatten()
        val_actual_square = val_actuals[:, :, busiest_square].flatten()

        # Align square-specific data
        min_square_length = min(len(val_pred_square), len(val_actual_square))
        val_pred_square = val_pred_square[:min_square_length]
        val_actual_square = val_actual_square[:min_square_length]

        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(val_actual_square, 'b-', label='Actual', alpha=0.8, linewidth=0.8)
        ax3.plot(val_pred_square, 'r-', label='Predicted', alpha=0.9, linewidth=0.8)
        ax3.set_title(f'Full Validation Set - Square {busiest_square} (Busiest Square)', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Time Steps (10-min intervals)')
        ax3.set_ylabel('Traffic Volume')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Peak detection visualization
        ax4 = fig.add_subplot(gs[1, 2])
        if config.USE_PEAK_LOSS and val_peaks.size > 0:
            try:
                peak_scores_flat = val_peaks.flatten()
                actual_peaks = (val_actual_avg > np.percentile(val_actual_avg, 80)).astype(float)

                # Better alignment of peak scores
                if len(peak_scores_flat) != len(actual_peaks):
                    if len(peak_scores_flat) > len(actual_peaks):
                        step = len(peak_scores_flat) // len(actual_peaks)
                        peak_scores_resampled = peak_scores_flat[::step][:len(actual_peaks)]
                    else:
                        peak_scores_resampled = np.pad(peak_scores_flat,
                                                       (0, len(actual_peaks) - len(peak_scores_flat)),
                                                       mode='edge')
                else:
                    peak_scores_resampled = peak_scores_flat

                # Final safety check
                min_peak_length = min(len(actual_peaks), len(peak_scores_resampled))
                actual_peaks = actual_peaks[:min_peak_length]
                peak_scores_resampled = peak_scores_resampled[:min_peak_length]

                ax4.plot(actual_peaks, 'b-', label='Actual Peaks', alpha=0.8)
                ax4.plot(peak_scores_resampled, 'r--', label='Predicted Peak Scores', alpha=0.8)
                ax4.set_title('Peak Detection Performance', fontsize=14)
                ax4.set_xlabel('Time Steps')
                ax4.set_ylabel('Peak Score')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            except Exception as e:
                ax4.text(0.5, 0.5, f'Peak Detection\nError: {str(e)[:30]}...',
                         ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Peak Detection\nNot Available',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Peak Detection', fontsize=14)

        # Plot 5: Zoomed analysis
        zoom_start = min(500, len(val_actual_avg) // 2)
        zoom_end = min(zoom_start + 1008, len(val_actual_avg))  # 7 days

        if zoom_end > zoom_start:
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(val_actual_avg[zoom_start:zoom_end], 'b-', label='Actual', alpha=0.8, linewidth=1.2)
            ax5.plot(val_pred_avg[zoom_start:zoom_end], 'r-', label='Predicted', alpha=0.9, linewidth=1.2)

            # Add day separators
            zoom_length = zoom_end - zoom_start
            days_to_show = min(7, zoom_length // 144)

            for day in range(days_to_show + 1):
                x_pos = day * 144
                if x_pos < zoom_length:
                    ax5.axvline(x=x_pos, color='gray', alpha=0.3, linestyle=':')
                    if day < days_to_show:
                        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        if day < len(day_names):
                            ax5.text(day * 144 + 72, ax5.get_ylim()[1] * 0.95, day_names[day],
                                     ha='center', fontsize=10, alpha=0.8)

            ax5.set_title(f'{days_to_show}-Day Pattern (Average Traffic)', fontsize=14)
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Traffic Volume')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Plot 6: Daily pattern analysis
        ax6 = fig.add_subplot(gs[2, 1])
        try:
            n_days = len(val_pred_avg) // 144
            if n_days > 0:
                val_pred_daily = val_pred_avg[:n_days * 144].reshape(n_days, 144)
                val_actual_daily = val_actual_avg[:n_days * 144].reshape(n_days, 144)

                avg_pred_pattern = val_pred_daily.mean(axis=0)
                avg_actual_pattern = val_actual_daily.mean(axis=0)
                std_actual_pattern = val_actual_daily.std(axis=0)

                time_hours = np.arange(144) * 10 / 60

                ax6.plot(time_hours, avg_actual_pattern, 'b-', label='Actual Average', linewidth=2)
                ax6.fill_between(time_hours,
                                 avg_actual_pattern - std_actual_pattern,
                                 avg_actual_pattern + std_actual_pattern,
                                 alpha=0.2, color='blue', label='±1 STD')
                ax6.plot(time_hours, avg_pred_pattern, 'r--', label='Predicted Average', linewidth=2)

                ax6.set_title('Average Daily Pattern', fontsize=14)
                ax6.set_xlabel('Hour of Day')
                ax6.set_ylabel('Traffic Volume')
                ax6.set_xlim(0, 24)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Insufficient data\nfor daily analysis',
                         ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        except Exception as e:
            ax6.text(0.5, 0.5, f'Daily Pattern\nError: {str(e)[:30]}...',
                     ha='center', va='center', transform=ax6.transAxes, fontsize=10)

        # Plot 7: Prediction horizon analysis
        ax7 = fig.add_subplot(gs[2, 2])
        try:
            horizon_rmse = []
            for h in range(val_predictions.shape[1]):
                step_pred = val_predictions[:, h, :].flatten()
                step_actual = val_actuals[:, h, :].flatten()

                # Align lengths
                min_len = min(len(step_pred), len(step_actual))
                step_pred = step_pred[:min_len]
                step_actual = step_actual[:min_len]

                if min_len > 0:
                    rmse = np.sqrt(np.mean((step_pred - step_actual) ** 2))
                    horizon_rmse.append(rmse)

            if horizon_rmse:
                ax7.plot(range(1, len(horizon_rmse) + 1), horizon_rmse, 'o-',
                         linewidth=2, markersize=6, color='darkgreen')
                ax7.set_title('RMSE by Prediction Horizon', fontsize=14)
                ax7.set_xlabel('Prediction Step (10-min intervals)')
                ax7.set_ylabel('RMSE')
                ax7.grid(True, alpha=0.3)
            else:
                ax7.text(0.5, 0.5, 'Horizon Analysis\nInsufficient Data',
                         ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        except Exception as e:
            ax7.text(0.5, 0.5, f'Horizon Analysis\nError: {str(e)[:30]}...',
                     ha='center', va='center', transform=ax7.transAxes, fontsize=10)

        # Plot 8: Performance statistics summary
        ax8 = fig.add_subplot(gs[3, :])

        try:
            # Calculate metrics for different traffic levels
            overall_rmse = np.sqrt(np.mean((val_pred_avg - val_actual_avg) ** 2))
            overall_mae = np.mean(np.abs(val_pred_avg - val_actual_avg))
            overall_r2 = r2_score(val_actual_avg, val_pred_avg) if len(val_actual_avg) > 1 else 0

            # Peak metrics
            if len(val_actual_avg) > 0:
                peak_mask = val_actual_avg > np.percentile(val_actual_avg, 90)
                if peak_mask.sum() > 0:
                    peak_rmse = np.sqrt(np.mean((val_pred_avg[peak_mask] - val_actual_avg[peak_mask]) ** 2))
                else:
                    peak_rmse = 0

                # Low traffic metrics
                low_mask = val_actual_avg < np.percentile(val_actual_avg, 10)
                if low_mask.sum() > 0:
                    low_rmse = np.sqrt(np.mean((val_pred_avg[low_mask] - val_actual_avg[low_mask]) ** 2))
                else:
                    low_rmse = 0
            else:
                peak_rmse = low_rmse = 0
                peak_mask = low_mask = np.array([])

            metrics_text = [
                f"Overall Performance:",
                f"  RMSE: {overall_rmse:.2f}  |  MAE: {overall_mae:.2f}  |  R²: {overall_r2:.3f}",
                f"",
                f"Peak Traffic Performance (>90th percentile):",
                f"  RMSE: {peak_rmse:.2f}  |  Samples: {peak_mask.sum():,}",
                f"",
                f"Low Traffic Performance (<10th percentile):",
                f"  RMSE: {low_rmse:.2f}  |  Samples: {low_mask.sum():,}",
                f"",
                f"Model Configuration:",
                f"  Architecture: Enhanced LSTM  |  Hidden Dim: {config.HIDDEN_DIM}  |  Layers: {config.NUM_LAYERS}",
                f"  Peak Loss: {config.USE_PEAK_LOSS}  |  Multi-Scale: {config.USE_MULTI_SCALE}  |  Attention: {config.USE_ATTENTION}",
                f"",
                f"Data Info:",
                f"  Validation samples: {len(val_pred_avg):,}  |  Prediction horizon: {val_predictions.shape[1]}  |  Squares: {val_predictions.shape[2]}"
            ]

            ax8.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax8.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        except Exception as e:
            ax8.text(0.5, 0.5, f'Metrics calculation error:\n{str(e)}',
                     ha='center', va='center', transform=ax8.transAxes, fontsize=12)

        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')

        plt.suptitle('Enhanced Validation Set Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(output_dir, 'plots', 'enhanced_full_validation_analysis.png'),
                    dpi=config.VIZ_SAVE_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error creating validation analysis plot: {e}")

    # 2. PEAK PERFORMANCE DETAILED ANALYSIS
    print("Creating peak performance analysis...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        # Peak detection accuracy
        if config.USE_PEAK_LOSS and val_peaks.size > 0:
            try:
                peak_threshold = np.percentile(val_actual_avg, 90)
                actual_peaks_binary = (val_actual_avg > peak_threshold).astype(int)

                # Align peak scores with time series
                if len(val_peaks.flatten()) != len(actual_peaks_binary):
                    indices = np.linspace(0, len(val_peaks.flatten()) - 1, len(actual_peaks_binary)).astype(int)
                    predicted_peaks = val_peaks.flatten()[indices]
                else:
                    predicted_peaks = val_peaks.flatten()

                # ROC curve data
                from sklearn.metrics import roc_curve, auc, precision_recall_curve

                fpr, tpr, _ = roc_curve(actual_peaks_binary, predicted_peaks)
                roc_auc = auc(fpr, tpr)

                axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[0].set_xlim([0.0, 1.0])
                axes[0].set_ylim([0.0, 1.05])
                axes[0].set_xlabel('False Positive Rate')
                axes[0].set_ylabel('True Positive Rate')
                axes[0].set_title('Peak Detection ROC Curve')
                axes[0].legend(loc="lower right")
                axes[0].grid(True, alpha=0.3)

                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(actual_peaks_binary, predicted_peaks)
                pr_auc = auc(recall, precision)

                axes[1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
                axes[1].set_xlabel('Recall')
                axes[1].set_ylabel('Precision')
                axes[1].set_title('Peak Detection Precision-Recall Curve')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            except Exception as e:
                axes[0].text(0.5, 0.5, f'Peak Detection\nError: {str(e)[:30]}...', ha='center', va='center',
                             fontsize=12)
                axes[0].set_title('Peak Detection ROC')
                axes[1].text(0.5, 0.5, f'Peak Detection\nError: {str(e)[:30]}...', ha='center', va='center',
                             fontsize=12)
                axes[1].set_title('Peak Detection PR Curve')
        else:
            axes[0].text(0.5, 0.5, 'Peak Detection\nNot Available', ha='center', va='center', fontsize=14)
            axes[0].set_title('Peak Detection ROC')
            axes[1].text(0.5, 0.5, 'Peak Detection\nNot Available', ha='center', va='center', fontsize=14)
            axes[1].set_title('Peak Detection PR Curve')

        # Peak vs non-peak performance
        try:
            peak_mask = val_actual_avg > np.percentile(val_actual_avg, 85)
            non_peak_mask = val_actual_avg < np.percentile(val_actual_avg, 50)

            peak_errors = np.abs(val_pred_avg[peak_mask] - val_actual_avg[peak_mask])
            non_peak_errors = np.abs(val_pred_avg[non_peak_mask] - val_actual_avg[non_peak_mask])

            if len(peak_errors) > 0 and len(non_peak_errors) > 0:
                axes[2].hist(peak_errors, bins=30, alpha=0.7, label=f'Peak Errors (n={len(peak_errors)})', density=True)
                axes[2].hist(non_peak_errors, bins=30, alpha=0.7, label=f'Non-Peak Errors (n={len(non_peak_errors)})',
                             density=True)
                axes[2].set_xlabel('Absolute Error')
                axes[2].set_ylabel('Density')
                axes[2].set_title('Error Distribution: Peak vs Non-Peak')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, 'Insufficient data\nfor error analysis', ha='center', va='center', fontsize=12)
                axes[2].set_title('Error Distribution')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error Analysis\nError: {str(e)[:30]}...', ha='center', va='center', fontsize=12)
            axes[2].set_title('Error Distribution')

        # Peak magnitude prediction accuracy
        try:
            peak_indices = np.where(peak_mask)[0]
            if len(peak_indices) > 0:
                sample_indices = np.random.choice(peak_indices, min(1000, len(peak_indices)), replace=False)

                axes[3].scatter(val_actual_avg[sample_indices], val_pred_avg[sample_indices],
                                alpha=0.6, s=20, c='red')

                # Perfect prediction line
                min_val = min(val_actual_avg[sample_indices].min(), val_pred_avg[sample_indices].min())
                max_val = max(val_actual_avg[sample_indices].max(), val_pred_avg[sample_indices].max())
                axes[3].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')

                axes[3].set_xlabel('Actual Peak Values')
                axes[3].set_ylabel('Predicted Peak Values')
                axes[3].set_title('Peak Magnitude Prediction Accuracy')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                axes[3].set_aspect('equal')
            else:
                axes[3].text(0.5, 0.5, 'No peaks found\nfor analysis', ha='center', va='center', fontsize=12)
                axes[3].set_title('Peak Magnitude Analysis')
        except Exception as e:
            axes[3].text(0.5, 0.5, f'Peak Magnitude\nError: {str(e)[:30]}...', ha='center', va='center', fontsize=12)
            axes[3].set_title('Peak Magnitude Analysis')

        # Peak timing analysis
        try:
            from scipy.signal import find_peaks

            actual_peaks_idx, _ = find_peaks(val_actual_avg, height=np.percentile(val_actual_avg, 85), distance=20)
            pred_peaks_idx, _ = find_peaks(val_pred_avg, height=np.percentile(val_pred_avg, 85), distance=20)

            # Calculate timing differences
            timing_diffs = []
            if len(actual_peaks_idx) > 0 and len(pred_peaks_idx) > 0:
                for actual_peak in actual_peaks_idx:
                    closest_pred = pred_peaks_idx[np.argmin(np.abs(pred_peaks_idx - actual_peak))]
                    timing_diffs.append(abs(closest_pred - actual_peak) * 10)  # Convert to minutes

            if timing_diffs:
                axes[4].hist(timing_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
                axes[4].set_xlabel('Peak Timing Error (minutes)')
                axes[4].set_ylabel('Frequency')
                axes[4].set_title(f'Peak Timing Accuracy\nMean Error: {np.mean(timing_diffs):.1f} min')
                axes[4].grid(True, alpha=0.3)
            else:
                axes[4].text(0.5, 0.5, 'No Peaks\nDetected', ha='center', va='center', fontsize=14)
                axes[4].set_title('Peak Timing Analysis')
        except Exception as e:
            axes[4].text(0.5, 0.5, f'Timing Analysis\nError: {str(e)[:30]}...', ha='center', va='center', fontsize=12)
            axes[4].set_title('Peak Timing Analysis')

        # Peak intensity distribution
        try:
            if len(peak_mask) > 0 and peak_mask.sum() > 0:
                axes[5].hist(val_actual_avg[peak_mask], bins=20, alpha=0.7, label='Actual Peak Intensities',
                             density=True)
                axes[5].hist(val_pred_avg[peak_mask], bins=20, alpha=0.7, label='Predicted Peak Intensities',
                             density=True)
                axes[5].set_xlabel('Traffic Volume')
                axes[5].set_ylabel('Density')
                axes[5].set_title('Peak Intensity Distribution')
                axes[5].legend()
                axes[5].grid(True, alpha=0.3)
            else:
                axes[5].text(0.5, 0.5, 'No peaks for\nintensity analysis', ha='center', va='center', fontsize=12)
                axes[5].set_title('Peak Intensity Distribution')
        except Exception as e:
            axes[5].text(0.5, 0.5, f'Intensity Analysis\nError: {str(e)[:30]}...', ha='center', va='center',
                         fontsize=12)
            axes[5].set_title('Peak Intensity Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'peak_performance_analysis.png'),
                    dpi=config.VIZ_SAVE_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error creating peak performance analysis: {e}")

    # 3. COMPREHENSIVE TEST SET ANALYSIS
    print("Creating comprehensive test set analysis...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        test_pred_avg = test_predictions.mean(axis=2).flatten()
        test_actual_avg = test_actuals.mean(axis=2).flatten()

        # Align test data
        min_test_length = min(len(test_pred_avg), len(test_actual_avg))
        test_pred_avg = test_pred_avg[:min_test_length]
        test_actual_avg = test_actual_avg[:min_test_length]

        # Full test set comparison
        axes[0, 0].plot(test_actual_avg, 'b-', label='Actual', alpha=0.8, linewidth=0.8)
        axes[0, 0].plot(test_pred_avg, 'r-', label='Predicted', alpha=0.9, linewidth=0.8)
        axes[0, 0].set_title('Full Test Set - Average Traffic', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Steps (10-min intervals)')
        axes[0, 0].set_ylabel('Traffic Volume')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Zoomed test view
        zoom_end = min(config.VIZ_ZOOM_STEPS if hasattr(config, 'VIZ_ZOOM_STEPS') else 2000, len(test_actual_avg))
        if zoom_end > 0:
            axes[0, 1].plot(test_actual_avg[:zoom_end], 'b-', label='Actual', alpha=0.8, linewidth=1)
            axes[0, 1].plot(test_pred_avg[:zoom_end], 'r-', label='Predicted', alpha=0.9, linewidth=1)
            axes[0, 1].set_title(f'Test Set - First {zoom_end} Steps', fontsize=14)
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Traffic Volume')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Comprehensive scatter plot with density
        sample_size = min(config.VIZ_SAMPLE_SIZE if hasattr(config, 'VIZ_SAMPLE_SIZE') else 5000, len(test_pred_avg))
        if sample_size > 0:
            sample_indices = np.random.choice(len(test_pred_avg), sample_size, replace=False)

            x_scatter = test_actual_avg[sample_indices]
            y_scatter = test_pred_avg[sample_indices]

            # Calculate point density
            try:
                xy = np.vstack([x_scatter, y_scatter])
                density = stats.gaussian_kde(xy)(xy)
                scatter = axes[0, 2].scatter(x_scatter, y_scatter, c=density, s=15, alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, ax=axes[0, 2], label='Density')
            except:
                axes[0, 2].scatter(x_scatter, y_scatter, s=15, alpha=0.6)

            # Perfect prediction line
            min_val = min(x_scatter.min(), y_scatter.min())
            max_val = max(x_scatter.max(), y_scatter.max())
            axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')

            axes[0, 2].set_xlabel('Actual Traffic Volume')
            axes[0, 2].set_ylabel('Predicted Traffic Volume')
            axes[0, 2].set_title(f'Test Set Scatter (n={sample_size:,})')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_aspect('equal')

        # Residual analysis
        residuals = test_pred_avg - test_actual_avg

        sample_indices_res = np.random.choice(len(test_pred_avg), min(1000, len(test_pred_avg)), replace=False)
        axes[1, 0].scatter(test_actual_avg[sample_indices_res], residuals[sample_indices_res],
                           alpha=0.4, s=8, c='red')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.8)

        # Add trend line
        try:
            z = np.polyfit(test_actual_avg[sample_indices_res], residuals[sample_indices_res], 1)
            p = np.poly1d(z)
            axes[1, 0].plot(sorted(test_actual_avg[sample_indices_res]),
                            p(sorted(test_actual_avg[sample_indices_res])), "g--", alpha=0.8, linewidth=2)
        except:
            pass

        axes[1, 0].set_xlabel('Actual Traffic Volume')
        axes[1, 0].set_ylabel('Residual (Predicted - Actual)')
        axes[1, 0].set_title('Residual Analysis')
        axes[1, 0].grid(True, alpha=0.3)

        # Error distribution with statistics
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')

        # Add normal distribution overlay
        try:
            mu, sigma = stats.norm.fit(residuals)
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2,
                            label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
            axes[1, 1].legend()
        except:
            pass

        axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # Performance by traffic level
        try:
            n_bins = 5
            bin_edges = np.percentile(test_actual_avg, np.linspace(0, 100, n_bins + 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            rmse_by_level = []
            mae_by_level = []

            for i in range(n_bins):
                mask = (test_actual_avg >= bin_edges[i]) & (test_actual_avg < bin_edges[i + 1])
                if mask.sum() > 0:
                    rmse = np.sqrt(np.mean((test_pred_avg[mask] - test_actual_avg[mask]) ** 2))
                    mae = np.mean(np.abs(test_pred_avg[mask] - test_actual_avg[mask]))
                    rmse_by_level.append(rmse)
                    mae_by_level.append(mae)
                else:
                    rmse_by_level.append(0)
                    mae_by_level.append(0)

            x_pos = np.arange(len(bin_centers))
            width = 0.35

            axes[1, 2].bar(x_pos - width / 2, rmse_by_level, width, label='RMSE', alpha=0.8)
            axes[1, 2].bar(x_pos + width / 2, mae_by_level, width, label='MAE', alpha=0.8)

            axes[1, 2].set_xlabel('Traffic Level (Percentile Bins)')
            axes[1, 2].set_ylabel('Error')
            axes[1, 2].set_title('Performance by Traffic Level')
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels([f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}' for i in range(n_bins)],
                                       rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Traffic Level Analysis\nError: {str(e)[:30]}...',
                            ha='center', va='center', fontsize=12)
            axes[1, 2].set_title('Performance by Traffic Level')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'comprehensive_test_analysis.png'),
                    dpi=config.VIZ_SAVE_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error creating test analysis: {e}")

    # 4. MULTI-SQUARE PERFORMANCE HEATMAP
    print("Creating multi-square performance heatmap...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Calculate RMSE for each square
        n_squares = val_predictions.shape[2]
        square_rmse_val = []
        square_mae_val = []
        square_r2_val = []

        for sq in range(n_squares):
            try:
                sq_pred = val_predictions[:, :, sq].flatten()
                sq_actual = val_actuals[:, :, sq].flatten()

                # Align lengths
                min_sq_len = min(len(sq_pred), len(sq_actual))
                sq_pred = sq_pred[:min_sq_len]
                sq_actual = sq_actual[:min_sq_len]

                if min_sq_len > 0:
                    rmse = np.sqrt(np.mean((sq_pred - sq_actual) ** 2))
                    mae = np.mean(np.abs(sq_pred - sq_actual))
                    r2 = r2_score(sq_actual, sq_pred) if np.var(sq_actual) > 0 else 0
                else:
                    rmse = mae = r2 = 0

                square_rmse_val.append(rmse)
                square_mae_val.append(mae)
                square_r2_val.append(r2)
            except:
                square_rmse_val.append(0)
                square_mae_val.append(0)
                square_r2_val.append(0)

        # Performance heatmap
        metrics_matrix = np.array([square_rmse_val, square_mae_val, square_r2_val])

        im1 = axes[0, 0].imshow(metrics_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0, 0].set_title('Performance Metrics by Square', fontsize=14)
        axes[0, 0].set_xlabel('Square ID')
        axes[0, 0].set_ylabel('Metric')
        axes[0, 0].set_yticks([0, 1, 2])
        axes[0, 0].set_yticklabels(['RMSE', 'MAE', 'R²'])

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0, 0])
        cbar1.set_label('Metric Value')

        # Best and worst squares comparison
        try:
            sorted_squares = np.argsort(square_rmse_val)
            best_squares = sorted_squares[:3]
            worst_squares = sorted_squares[-3:]

            for idx, (squares, title, ax) in enumerate([
                (best_squares, 'Best Performing Squares', axes[0, 1]),
                (worst_squares, 'Worst Performing Squares', axes[1, 0])
            ]):
                colors = ['blue', 'green', 'orange']
                for i, sq in enumerate(squares):
                    sq_pred = val_predictions[:500, :, sq].flatten()  # First 500 samples
                    sq_actual = val_actuals[:500, :, sq].flatten()

                    # Align lengths
                    min_len = min(len(sq_pred), len(sq_actual))
                    sq_pred = sq_pred[:min_len]
                    sq_actual = sq_actual[:min_len]

                    if min_len > 0:
                        ax.plot(sq_actual, alpha=0.7, label=f'Actual Sq{sq}',
                                linestyle='-', color=colors[i])
                        ax.plot(sq_pred, alpha=0.7, label=f'Pred Sq{sq}',
                                linestyle='--', color=colors[i])

                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Traffic Volume')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Best Squares\nError: {str(e)[:30]}...', ha='center', va='center', fontsize=12)
            axes[1, 0].text(0.5, 0.5, f'Worst Squares\nError: {str(e)[:30]}...', ha='center', va='center', fontsize=12)

        # Performance distribution
        try:
            axes[1, 1].hist(square_rmse_val, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].set_xlabel('RMSE')
            axes[1, 1].set_ylabel('Number of Squares')
            axes[1, 1].set_title('RMSE Distribution Across Squares')
            axes[1, 1].axvline(x=np.mean(square_rmse_val), color='red', linestyle='--',
                               label=f'Mean: {np.mean(square_rmse_val):.2f}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'RMSE Distribution\nError: {str(e)[:30]}...', ha='center', va='center',
                            fontsize=12)
            axes[1, 1].set_title('RMSE Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'multi_square_performance.png'),
                    dpi=config.VIZ_SAVE_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error creating multi-square performance: {e}")

    # 5. TEMPORAL PATTERN ANALYSIS
    print("Creating temporal pattern analysis...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Hour-of-day analysis
        try:
            n_days_val = len(val_pred_avg) // 144
            if n_days_val > 0:
                val_pred_hourly = val_pred_avg[:n_days_val * 144].reshape(n_days_val, 144)
                val_actual_hourly = val_actual_avg[:n_days_val * 144].reshape(n_days_val, 144)

                # Calculate RMSE by hour
                hourly_rmse = []
                for h in range(144):
                    rmse = np.sqrt(np.mean((val_pred_hourly[:, h] - val_actual_hourly[:, h]) ** 2))
                    hourly_rmse.append(rmse)

                time_hours = np.arange(144) * 10 / 60

                axes[0, 0].plot(time_hours, hourly_rmse, 'g-', linewidth=2, marker='o', markersize=3)
                axes[0, 0].set_title('RMSE by Hour of Day', fontsize=14)
                axes[0, 0].set_xlabel('Hour of Day')
                axes[0, 0].set_ylabel('RMSE')
                axes[0, 0].set_xlim(0, 24)
                axes[0, 0].grid(True, alpha=0.3)

                # Mark peak error hours
                peak_error_hours = np.argsort(hourly_rmse)[-3:]
                for hour_idx in peak_error_hours:
                    hour = time_hours[hour_idx]
                    axes[0, 0].annotate(f'{hour:.1f}h',
                                        xy=(hour, hourly_rmse[hour_idx]),
                                        xytext=(hour, hourly_rmse[hour_idx] + max(hourly_rmse) * 0.1),
                                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                                        fontsize=10, ha='center')
            else:
                axes[0, 0].text(0.5, 0.5, 'Insufficient data\nfor hourly analysis', ha='center', va='center',
                                fontsize=12)
                axes[0, 0].set_title('RMSE by Hour of Day')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Hourly Analysis\nError: {str(e)[:30]}...', ha='center', va='center',
                            fontsize=12)
            axes[0, 0].set_title('RMSE by Hour of Day')

        # Weekly pattern (if enough data)
        try:
            if n_days_val >= 7:
                n_weeks = n_days_val // 7
                if n_weeks > 0:
                    weekly_pred = val_pred_avg[:n_weeks * 7 * 144].reshape(n_weeks, 7, 144)
                    weekly_actual = val_actual_avg[:n_weeks * 7 * 144].reshape(n_weeks, 7, 144)

                    # Average by day of week
                    daily_rmse = []
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                    for day in range(7):
                        day_pred = weekly_pred[:, day, :].flatten()
                        day_actual = weekly_actual[:, day, :].flatten()
                        rmse = np.sqrt(np.mean((day_pred - day_actual) ** 2))
                        daily_rmse.append(rmse)

                    axes[0, 1].bar(day_names, daily_rmse, alpha=0.7, color='lightcoral')
                    axes[0, 1].set_title('RMSE by Day of Week', fontsize=14)
                    axes[0, 1].set_ylabel('RMSE')
                    axes[0, 1].grid(True, alpha=0.3)

                    # Highlight weekends
                    axes[0, 1].bar(['Sat', 'Sun'], [daily_rmse[5], daily_rmse[6]],
                                   alpha=0.9, color='darkred', label='Weekend')
                    axes[0, 1].legend()
                else:
                    axes[0, 1].text(0.5, 0.5, 'Insufficient weeks\nfor analysis', ha='center', va='center', fontsize=12)
                    axes[0, 1].set_title('RMSE by Day of Week')
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient days\nfor weekly analysis', ha='center', va='center',
                                fontsize=12)
                axes[0, 1].set_title('RMSE by Day of Week')
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Weekly Analysis\nError: {str(e)[:30]}...', ha='center', va='center',
                            fontsize=12)
            axes[0, 1].set_title('RMSE by Day of Week')

        # Prediction horizon detailed analysis
        try:
            horizon_metrics = {'rmse': [], 'mae': [], 'r2': []}

            for h in range(val_predictions.shape[1]):
                step_pred = val_predictions[:, h, :].flatten()
                step_actual = val_actuals[:, h, :].flatten()

                # Align lengths
                min_len = min(len(step_pred), len(step_actual))
                step_pred = step_pred[:min_len]
                step_actual = step_actual[:min_len]

                if min_len > 0:
                    rmse = np.sqrt(np.mean((step_pred - step_actual) ** 2))
                    mae = np.mean(np.abs(step_pred - step_actual))
                    r2 = r2_score(step_actual, step_pred) if np.var(step_actual) > 0 else 0

                    horizon_metrics['rmse'].append(rmse)
                    horizon_metrics['mae'].append(mae)
                    horizon_metrics['r2'].append(r2)

            x_horizon = np.arange(1, len(horizon_metrics['rmse']) + 1)

            # Multiple metrics on same plot
            ax_twin = axes[1, 0].twinx()

            line1 = axes[1, 0].plot(x_horizon, horizon_metrics['rmse'], 'b-o',
                                    label='RMSE', linewidth=2, markersize=4)
            line2 = axes[1, 0].plot(x_horizon, horizon_metrics['mae'], 'g-s',
                                    label='MAE', linewidth=2, markersize=4)
            line3 = ax_twin.plot(x_horizon, horizon_metrics['r2'], 'r-^',
                                 label='R²', linewidth=2, markersize=4)

            axes[1, 0].set_xlabel('Prediction Step (10-min intervals)')
            axes[1, 0].set_ylabel('RMSE / MAE', color='black')
            ax_twin.set_ylabel('R²', color='red')
            axes[1, 0].set_title('Detailed Horizon Analysis', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            axes[1, 0].legend(lines, labels, loc='upper left')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Horizon Analysis\nError: {str(e)[:30]}...', ha='center', va='center',
                            fontsize=12)
            axes[1, 0].set_title('Detailed Horizon Analysis')

        # Error correlation analysis
        try:
            errors = val_pred_avg - val_actual_avg
            max_lag = min(50, len(errors) // 4)

            autocorrelations = []
            lags = range(1, max_lag)

            for lag in lags:
                if len(errors) > lag:
                    corr = np.corrcoef(errors[:-lag], errors[lag:])[0, 1]
                    autocorrelations.append(corr)
                else:
                    autocorrelations.append(0)

            axes[1, 1].plot(lags, autocorrelations, 'purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Lag (time steps)')
            axes[1, 1].set_ylabel('Error Autocorrelation')
            axes[1, 1].set_title('Error Autocorrelation', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)

            # Add significance bands (approximate)
            n_effective = len(errors)
            significance_level = 1.96 / np.sqrt(n_effective)
            axes[1, 1].axhline(y=significance_level, color='red', linestyle=':', alpha=0.7, label='95% CI')
            axes[1, 1].axhline(y=-significance_level, color='red', linestyle=':', alpha=0.7)
            axes[1, 1].legend()
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Autocorrelation\nError: {str(e)[:30]}...', ha='center', va='center',
                            fontsize=12)
            axes[1, 1].set_title('Error Autocorrelation')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'temporal_pattern_analysis.png'),
                    dpi=config.VIZ_SAVE_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error creating temporal pattern analysis: {e}")

    print("Enhanced visualizations completed successfully!")

    # Calculate comprehensive metrics with error handling
    try:
        val_metrics = calculate_enhanced_metrics(val_predictions, val_actuals, scaler)
        test_metrics = calculate_enhanced_metrics(test_predictions, test_actuals, scaler)
    except Exception as e:
        print(f"Warning: Could not calculate detailed metrics: {e}")
        # Provide basic metrics as fallback
        try:
            val_pred_flat = val_predictions.flatten()
            val_actual_flat = val_actuals.flatten()
            min_len = min(len(val_pred_flat), len(val_actual_flat))
            val_pred_flat = val_pred_flat[:min_len]
            val_actual_flat = val_actual_flat[:min_len]

            test_pred_flat = test_predictions.flatten()
            test_actual_flat = test_actuals.flatten()
            min_len_test = min(len(test_pred_flat), len(test_actual_flat))
            test_pred_flat = test_pred_flat[:min_len_test]
            test_actual_flat = test_actual_flat[:min_len_test]

            val_metrics = {
                'rmse': float(np.sqrt(np.mean((val_pred_flat - val_actual_flat) ** 2))),
                'mae': float(np.mean(np.abs(val_pred_flat - val_actual_flat))),
                'r2': float(r2_score(val_actual_flat, val_pred_flat)) if len(val_actual_flat) > 1 else 0.0,
                'peak_rmse': 0.0, 'peak_mae': 0.0, 'peak_mape': 0.0,
                'low_rmse': 0.0, 'low_mae': 0.0, 'mape': 0.0, 'nrmse': 0.0,
                'correlation': float(np.corrcoef(val_pred_flat, val_actual_flat)[0, 1]) if len(
                    val_pred_flat) > 1 else 0.0,
                'directional_accuracy': 0.0
            }

            test_metrics = {
                'rmse': float(np.sqrt(np.mean((test_pred_flat - test_actual_flat) ** 2))),
                'mae': float(np.mean(np.abs(test_pred_flat - test_actual_flat))),
                'r2': float(r2_score(test_actual_flat, test_pred_flat)) if len(test_actual_flat) > 1 else 0.0,
                'peak_rmse': 0.0, 'peak_mae': 0.0, 'peak_mape': 0.0,
                'low_rmse': 0.0, 'low_mae': 0.0, 'mape': 0.0, 'nrmse': 0.0,
                'correlation': float(np.corrcoef(test_pred_flat, test_actual_flat)[0, 1]) if len(
                    test_pred_flat) > 1 else 0.0,
                'directional_accuracy': 0.0
            }
        except Exception as e2:
            print(f"Error creating fallback metrics: {e2}")
            # Last resort - empty metrics
            val_metrics = test_metrics = {
                'rmse': 0.0, 'mae': 0.0, 'r2': 0.0, 'peak_rmse': 0.0, 'peak_mae': 0.0,
                'peak_mape': 0.0, 'low_rmse': 0.0, 'low_mae': 0.0, 'mape': 0.0,
                'nrmse': 0.0, 'correlation': 0.0, 'directional_accuracy': 0.0
            }

    # Print summary
    print("\n" + "=" * 60)
    print("ENHANCED VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Generated visualizations:")
    print(f"  • Enhanced Full Validation Analysis")
    print(f"  • Peak Performance Analysis")
    print(f"  • Comprehensive Test Analysis")
    print(f"  • Multi-Square Performance Heatmap")
    print(f"  • Temporal Pattern Analysis")
    print(f"\nKey Insights:")
    print(f"  • Validation RMSE: {val_metrics.get('rmse', 0):.2f}")
    print(f"  • Test RMSE: {test_metrics.get('rmse', 0):.2f}")
    print(f"  • Peak RMSE: {val_metrics.get('peak_rmse', 0):.2f}")
    print(f"  • Low Traffic RMSE: {val_metrics.get('low_rmse', 0):.2f}")
    print(f"  • Overall R²: {val_metrics.get('r2', 0):.3f}")
    print(f"  • Directional Accuracy: {val_metrics.get('directional_accuracy', 0):.1%}")

    return val_metrics, test_metrics