"""
Training loop for forgery localization network
Implements chunked training for RAM constraints
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import csv

from ..models import get_model, get_loss_function
from ..data import get_dataset
from .metrics import MetricsTracker, EarlyStopping


class Trainer:
    """
    Trainer for forgery localization network
    Supports chunked training for large datasets (DocTamper)
    """
    
    def __init__(self, config, dataset_name: str = 'doctamper'):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
            dataset_name: Dataset to train on
        """
        self.config = config
        self.dataset_name = dataset_name
        
        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.get('system.device') == 'cuda' 
            else 'cpu'
        )
        print(f"Training on: {self.device}")
        
        # Initialize model
        self.model = get_model(config).to(self.device)
        
        # Loss function (dataset-aware)
        self.criterion = get_loss_function(config)
        
        # Optimizer
        lr = config.get('training.learning_rate', 0.001)
        weight_decay = config.get('training.weight_decay', 0.0001)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        epochs = config.get('training.epochs', 50)
        warmup_epochs = config.get('training.scheduler.warmup_epochs', 5)
        min_lr = config.get('training.scheduler.min_lr', 1e-5)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=epochs - warmup_epochs,
            T_mult=1,
            eta_min=min_lr
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics_tracker = MetricsTracker(config)
        
        # Early stopping
        patience = config.get('training.early_stopping.patience', 10)
        min_delta = config.get('training.early_stopping.min_delta', 0.001)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        # Output directories
        self.checkpoint_dir = Path(config.get('outputs.checkpoints', 'outputs/checkpoints'))
        self.log_dir = Path(config.get('outputs.logs', 'outputs/logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
    
    def create_dataloaders(self, 
                          chunk_start: float = 0.0,
                          chunk_end: float = 1.0) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders
        
        Args:
            chunk_start: Start ratio for chunked training
            chunk_end: End ratio for chunked training
        
        Returns:
            Train and validation dataloaders
        """
        batch_size = self.config.get('data.batch_size', 8)
        num_workers = self.config.get('system.num_workers', 4)
        
        # Training dataset (with chunking for DocTamper)
        if self.dataset_name == 'doctamper':
            train_dataset = get_dataset(
                self.config, 
                self.dataset_name, 
                split='train',
                chunk_start=chunk_start,
                chunk_end=chunk_end
            )
        else:
            train_dataset = get_dataset(
                self.config, 
                self.dataset_name, 
                split='train'
            )
        
        # Validation dataset (always full)
        # For FCD and SCD, validate on DocTamper TestingSet
        if self.dataset_name in ['fcd', 'scd']:
            val_dataset = get_dataset(
                self.config,
                'doctamper',  # Use DocTamper for validation
                split='val'
            )
        else:
            val_dataset = get_dataset(
                self.config,
                self.dataset_name,
                split='val' if self.dataset_name in ['doctamper', 'receipts'] else 'test'
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.get('system.pin_memory', True),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training dataloader
        
        Returns:
            Average loss and metrics
        """
        self.model.train()
        self.metrics_tracker.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, masks, metadata) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast():
                outputs, _ = self.model(images)
                
                # Dataset-aware loss
                has_pixel_mask = self.config.has_pixel_mask(self.dataset_name)
                losses = self.criterion.combined_loss(outputs, masks, has_pixel_mask)
            
            # Backward pass with gradient scaling
            self.scaler.scale(losses['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                self.metrics_tracker.update_segmentation(
                    probs, masks, self.dataset_name
                )
            
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'bce': f"{losses['bce'].item():.4f}"
            })
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute_all()
        
        return avg_loss, metrics
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate model
        
        Args:
            dataloader: Validation dataloader
        
        Returns:
            Average loss and metrics
        """
        self.model.eval()
        self.metrics_tracker.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for images, masks, metadata in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(images)
                
                # Dataset-aware loss
                has_pixel_mask = self.config.has_pixel_mask(self.dataset_name)
                losses = self.criterion.combined_loss(outputs, masks, has_pixel_mask)
                
                # Update metrics
                probs = torch.sigmoid(outputs)
                self.metrics_tracker.update_segmentation(
                    probs, masks, self.dataset_name
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}"
                })
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute_all()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, 
                       filename: str,
                       is_best: bool = False,
                       chunk_id: Optional[int] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'dataset': self.dataset_name,
            'chunk_id': chunk_id
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / f'best_{self.dataset_name}.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, filename: str, reset_epoch: bool = False):
        """
        Load model checkpoint
        
        Args:
            filename: Checkpoint filename
            reset_epoch: If True, reset epoch counter to 0 (useful for chunked training)
        """
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if reset_epoch:
            self.current_epoch = 0
            print(f"Loaded checkpoint: {path} (epoch counter reset to 0)")
        else:
            self.current_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
            print(f"Loaded checkpoint: {path} (resuming from epoch {self.current_epoch})")
        
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        return True
    
    def train(self,
              epochs: Optional[int] = None,
              chunk_start: float = 0.0,
              chunk_end: float = 1.0,
              chunk_id: Optional[int] = None,
              resume_from: Optional[str] = None):
        """
        Main training loop
        
        Args:
            epochs: Number of epochs (None uses config)
            chunk_start: Start ratio for chunked training
            chunk_end: End ratio for chunked training
            chunk_id: Chunk identifier for logging
            resume_from: Checkpoint to resume from
        """
        if epochs is None:
            epochs = self.config.get('training.epochs', 50)
        
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(chunk_start, chunk_end)
        
        print(f"\n{'='*60}")
        print(f"Training: {self.dataset_name}")
        if chunk_id is not None:
            print(f"Chunk: {chunk_id} [{chunk_start*100:.0f}% - {chunk_end*100:.0f}%]")
        print(f"Epochs: {epochs}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        # Training log file
        log_file = self.log_dir / f'{self.dataset_name}_chunk{chunk_id or 0}_log.csv'
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 
                           'train_iou', 'val_iou', 'train_dice', 'val_dice',
                           'train_precision', 'val_precision', 
                           'train_recall', 'val_recall', 'lr'])
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.metrics_tracker.log_epoch(epoch, 'train', train_loss, train_metrics)
            self.metrics_tracker.log_epoch(epoch, 'val', val_loss, val_metrics)
            
            # Log to file
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{train_metrics.get('iou', 0):.4f}",
                    f"{val_metrics.get('iou', 0):.4f}",
                    f"{train_metrics.get('dice', 0):.4f}",
                    f"{val_metrics.get('dice', 0):.4f}",
                    f"{train_metrics.get('precision', 0):.4f}",
                    f"{val_metrics.get('precision', 0):.4f}",
                    f"{train_metrics.get('recall', 0):.4f}",
                    f"{val_metrics.get('recall', 0):.4f}",
                    f"{current_lr:.6f}"
                ])
            
            # Print summary
            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train IoU: {train_metrics.get('iou', 0):.4f} | Val IoU: {val_metrics.get('iou', 0):.4f}")
            print(f"  Train Dice: {train_metrics.get('dice', 0):.4f} | Val Dice: {val_metrics.get('dice', 0):.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoints
            if self.config.get('training.checkpoint.save_every', 5) > 0:
                if (epoch + 1) % self.config.get('training.checkpoint.save_every', 5) == 0:
                    self.save_checkpoint(
                        f'{self.dataset_name}_chunk{chunk_id or 0}_epoch{epoch}.pth',
                        chunk_id=chunk_id
                    )
            
            # Check for best model
            monitor_metric = val_metrics.get('dice', 0)
            if monitor_metric > self.best_metric:
                self.best_metric = monitor_metric
                self.save_checkpoint(
                    f'{self.dataset_name}_chunk{chunk_id or 0}_best.pth',
                    is_best=True,
                    chunk_id=chunk_id
                )
            
            # Early stopping
            if self.early_stopping(monitor_metric):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save final checkpoint
        self.save_checkpoint(
            f'{self.dataset_name}_chunk{chunk_id or 0}_final.pth',
            chunk_id=chunk_id
        )
        
        # Save training history
        history_file = self.log_dir / f'{self.dataset_name}_chunk{chunk_id or 0}_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.metrics_tracker.get_history(), f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best Dice: {self.best_metric:.4f}")
        
        return self.metrics_tracker.get_history()


def get_trainer(config, dataset_name: str = 'doctamper') -> Trainer:
    """Factory function for trainer"""
    return Trainer(config, dataset_name)
