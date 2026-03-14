"""
Plotting utilities for training metrics visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json


def set_style():
    """Set matplotlib style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_training_curves(history: Dict, 
                        save_path: str,
                        title: str = "Training Progress"):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
        title: Plot title
    """
    set_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss
    ax = axes[0, 0]
    if 'train_loss' in history and history['train_loss']:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[0, 1]
    if 'train_iou' in history and history['train_iou']:
        ax.plot(epochs, history['train_iou'], 'b-', label='Train', linewidth=2)
    if 'val_iou' in history and history['val_iou']:
        ax.plot(epochs, history['val_iou'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dice
    ax = axes[0, 2]
    if 'train_dice' in history and history['train_dice']:
        ax.plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
    if 'val_dice' in history and history['val_dice']:
        ax.plot(epochs, history['val_dice'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Dice Score (F1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[1, 0]
    if 'train_precision' in history and history['train_precision']:
        ax.plot(epochs, history['train_precision'], 'b-', label='Train', linewidth=2)
    if 'val_precision' in history and history['val_precision']:
        ax.plot(epochs, history['val_precision'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 1]
    if 'train_recall' in history and history['train_recall']:
        ax.plot(epochs, history['train_recall'], 'b-', label='Train', linewidth=2)
    if 'val_recall' in history and history['val_recall']:
        ax.plot(epochs, history['val_recall'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary metrics bar chart
    ax = axes[1, 2]
    if history.get('val_iou') and history.get('val_dice'):
        metrics = ['IoU', 'Dice', 'Precision', 'Recall']
        final_values = [
            history['val_iou'][-1] if history['val_iou'] else 0,
            history['val_dice'][-1] if history['val_dice'] else 0,
            history['val_precision'][-1] if history.get('val_precision') else 0,
            history['val_recall'][-1] if history.get('val_recall') else 0
        ]
        colors = sns.color_palette("husl", 4)
        bars = ax.bar(metrics, final_values, color=colors)
        ax.set_ylabel('Score')
        ax.set_title('Final Validation Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, final_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         save_path: str,
                         title: str = "Confusion Matrix"):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        save_path: Path to save plot
        title: Plot title
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(importance: List[tuple],
                           save_path: str,
                           title: str = "Feature Importance"):
    """
    Plot feature importance
    
    Args:
        importance: List of (feature_name, importance) tuples
        save_path: Path to save plot
        title: Plot title
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    names = [item[0] for item in importance]
    values = [item[1] for item in importance]
    
    colors = sns.color_palette("viridis", len(importance))
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance saved to {save_path}")


def plot_dataset_comparison(all_histories: Dict[str, Dict],
                           save_path: str):
    """
    Plot comparison across datasets
    
    Args:
        all_histories: Dictionary of {dataset_name: history}
        save_path: Path to save plot
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['val_dice', 'val_iou']
    titles = ['Validation Dice Score', 'Validation IoU']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for dataset_name, history in all_histories.items():
            if metric in history and history[metric]:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=dataset_name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('val_', '').replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset comparison saved to {save_path}")


def plot_chunked_training_progress(chunk_histories: List[Dict],
                                   save_path: str,
                                   title: str = "Chunked Training Progress"):
    """
    Plot progress across training chunks
    
    Args:
        chunk_histories: List of history dictionaries per chunk
        save_path: Path to save plot
        title: Plot title
    """
    set_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = sns.color_palette("husl", len(chunk_histories))
    
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_dice', 'val_dice', 'Dice Score'),
        ('train_iou', 'val_iou', 'IoU'),
        ('train_precision', 'val_precision', 'Precision')
    ]
    
    for ax, (train_key, val_key, ylabel) in zip(axes.flat, metrics):
        total_epochs = 0
        
        for i, history in enumerate(chunk_histories):
            if train_key in history and history[train_key]:
                epochs = range(total_epochs + 1, total_epochs + len(history[train_key]) + 1)
                ax.plot(epochs, history[train_key], '--', color=colors[i], alpha=0.5)
                total_epochs += len(history[train_key])
        
        total_epochs = 0
        for i, history in enumerate(chunk_histories):
            if val_key in history and history[val_key]:
                epochs = range(total_epochs + 1, total_epochs + len(history[val_key]) + 1)
                ax.plot(epochs, history[val_key], '-', color=colors[i], 
                       label=f'Chunk {i+1}', linewidth=2)
                
                # Add vertical line for chunk boundary
                if i < len(chunk_histories) - 1:
                    ax.axvline(x=total_epochs + len(history[val_key]), 
                              color='gray', linestyle=':', alpha=0.5)
                
                total_epochs += len(history[val_key])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Validation {ylabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chunked training progress saved to {save_path}")


def generate_training_report(history: Dict,
                            save_path: str,
                            dataset_name: str = "unknown"):
    """
    Generate training report as text file
    
    Args:
        history: Training history
        save_path: Path to save report
        dataset_name: Dataset name
    """
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Training Report - {dataset_name}\n")
        f.write("="*60 + "\n\n")
        
        num_epochs = len(history.get('train_loss', []))
        f.write(f"Total Epochs: {num_epochs}\n\n")
        
        f.write("Final Metrics:\n")
        f.write("-"*40 + "\n")
        
        for key, values in history.items():
            if values and isinstance(values, list):
                final_value = values[-1]
                if isinstance(final_value, (int, float)):
                    f.write(f"  {key}: {final_value:.4f}\n")
        
        f.write("\n")
        f.write("Best Metrics:\n")
        f.write("-"*40 + "\n")
        
        for key, values in history.items():
            if values and isinstance(values, list):
                if 'loss' in key:
                    best_value = min(values)
                    best_epoch = values.index(best_value) + 1
                else:
                    best_value = max(values)
                    best_epoch = values.index(best_value) + 1
                
                if isinstance(best_value, (int, float)):
                    f.write(f"  {key}: {best_value:.4f} (epoch {best_epoch})\n")
    
    print(f"Training report saved to {save_path}")
