# multi_architecture_training/training_utils/training_monitor.py
# Training monitoring and early stopping utilities

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and implement early stopping"""
    
    def __init__(self, 
                 patience: int = 20,
                 min_delta: float = 1e-6,
                 restore_best_weights: bool = True,
                 monitor: str = 'val_loss'):
        """
        Initialize training monitor
        
        Args:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in monitored metric to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            monitor: Metric to monitor ('val_loss', 'train_loss', 'val_acc')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               model: torch.nn.Module, lr: Optional[float] = None) -> bool:
        """
        Update monitor with new epoch results
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            model: Model to save weights from
            lr: Learning rate (optional)
            
        Returns:
            bool: True if training should continue, False if should stop
        """
        # Store history
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        
        # Determine metric to monitor
        if self.monitor == 'val_loss':
            current_score = -val_loss  # Negative because we want to minimize loss
        elif self.monitor == 'train_loss':
            current_score = -train_loss
        else:
            current_score = val_loss  # Assume higher is better for other metrics
        
        # Check if this is the best score so far
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
            logger.info(f"🎯 New best {self.monitor}: {current_score:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            logger.info(f"⏳ No improvement for {self.counter} epochs (best: {self.best_score:.6f})")
        
        # Check if we should stop
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"🛑 Early stopping triggered after {self.patience} epochs without improvement")
            return False
            
        return True
    
    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Restore model to best weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"✅ Restored best weights from epoch {self.best_epoch}")
        else:
            logger.warning("⚠️  No best weights to restore")
    
    def get_training_summary(self) -> Dict:
        """Get training summary statistics"""
        if not self.train_losses:
            return {}
            
        return {
            'total_epochs': len(self.train_losses),
            'best_epoch': self.best_epoch,
            'best_val_loss': -self.best_score if self.monitor == 'val_loss' else self.best_score,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'early_stopped': self.should_stop,
            'improvement_epochs': self.counter
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation curves"""
        if not self.train_losses:
            logger.warning("⚠️  No training data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=self.best_epoch, color='g', linestyle='--', 
                   label=f'Best Epoch ({self.best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate curve (if available)
        if self.learning_rates:
            ax2.plot(epochs, self.learning_rates, 'purple', label='Learning Rate', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No learning rate data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training curves saved to {save_path}")
        
        plt.show()

class ConvergenceChecker:
    """Check for training convergence patterns"""
    
    @staticmethod
    def check_overfitting(train_losses: List[float], val_losses: List[float], 
                         window: int = 5) -> Dict:
        """
        Check for overfitting patterns
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            window: Window size for trend analysis
            
        Returns:
            Dict with overfitting analysis
        """
        if len(train_losses) < window * 2:
            return {'overfitting': False, 'confidence': 'low', 'reason': 'insufficient_data'}
        
        # Calculate recent trends
        recent_train_trend = np.mean(np.diff(train_losses[-window:]))
        recent_val_trend = np.mean(np.diff(val_losses[-window:]))
        
        # Calculate overall trends
        overall_train_trend = np.mean(np.diff(train_losses))
        overall_val_trend = np.mean(np.diff(val_losses))
        
        # Check for overfitting indicators
        overfitting_indicators = []
        
        # 1. Validation loss increasing while training loss decreasing
        if recent_train_trend < 0 and recent_val_trend > 0:
            overfitting_indicators.append('val_increasing_while_train_decreasing')
        
        # 2. Large gap between train and val loss
        recent_train_avg = np.mean(train_losses[-window:])
        recent_val_avg = np.mean(val_losses[-window:])
        gap_ratio = recent_val_avg / recent_train_avg if recent_train_avg > 0 else 1
        
        if gap_ratio > 1.5:
            overfitting_indicators.append('large_train_val_gap')
        
        # 3. Validation loss stopped improving
        if recent_val_trend > -1e-6:
            overfitting_indicators.append('val_not_improving')
        
        is_overfitting = len(overfitting_indicators) >= 2
        
        return {
            'overfitting': is_overfitting,
            'confidence': 'high' if len(overfitting_indicators) >= 2 else 'medium',
            'indicators': overfitting_indicators,
            'recent_train_trend': recent_train_trend,
            'recent_val_trend': recent_val_trend,
            'gap_ratio': gap_ratio
        }
    
    @staticmethod
    def check_convergence(losses: List[float], window: int = 10, 
                         threshold: float = 1e-6) -> Dict:
        """
        Check if training has converged
        
        Args:
            losses: List of losses to check
            window: Window size for convergence check
            threshold: Minimum change threshold
            
        Returns:
            Dict with convergence analysis
        """
        if len(losses) < window:
            return {'converged': False, 'confidence': 'low', 'reason': 'insufficient_data'}
        
        recent_losses = losses[-window:]
        loss_variance = np.var(recent_losses)
        loss_trend = np.mean(np.diff(recent_losses))
        
        converged = loss_variance < threshold and abs(loss_trend) < threshold
        
        return {
            'converged': converged,
            'confidence': 'high' if converged else 'medium',
            'loss_variance': loss_variance,
            'loss_trend': loss_trend,
            'threshold': threshold
        }

def create_training_report(monitor: TrainingMonitor, 
                          model_name: str,
                          save_dir: str) -> str:
    """
    Create a comprehensive training report
    
    Args:
        monitor: TrainingMonitor instance
        model_name: Name of the model
        save_dir: Directory to save report
        
    Returns:
        Path to saved report
    """
    summary = monitor.get_training_summary()
    
    # Create report content
    report_lines = [
        f"# Training Report: {model_name}",
        f"Generated: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Training Summary",
        f"- Total Epochs: {summary.get('total_epochs', 'N/A')}",
        f"- Best Epoch: {summary.get('best_epoch', 'N/A')}",
        f"- Best Validation Loss: {summary.get('best_val_loss', 'N/A'):.6f}",
        f"- Final Training Loss: {summary.get('final_train_loss', 'N/A'):.6f}",
        f"- Final Validation Loss: {summary.get('final_val_loss', 'N/A'):.6f}",
        f"- Early Stopped: {summary.get('early_stopped', 'N/A')}",
        "",
        "## Convergence Analysis",
    ]
    
    # Add convergence analysis
    if monitor.val_losses:
        convergence = ConvergenceChecker.check_convergence(monitor.val_losses)
        overfitting = ConvergenceChecker.check_overfitting(monitor.train_losses, monitor.val_losses)
        
        report_lines.extend([
            f"- Converged: {convergence['converged']}",
            f"- Overfitting: {overfitting['overfitting']}",
            f"- Confidence: {convergence['confidence']}",
            ""
        ])
    
    # Save report
    report_path = Path(save_dir) / f"{model_name}_training_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"📄 Training report saved to {report_path}")
    return str(report_path) 