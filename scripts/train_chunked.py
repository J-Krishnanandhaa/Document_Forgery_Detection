"""
Chunked Training Script for Document Forgery Detection

Supports training on large datasets (DocTamper) in chunks to manage RAM constraints.
Usage:
    python scripts/train_chunked.py --dataset doctamper --chunk 1
    python scripts/train_chunked.py --dataset rtm
    python scripts/train_chunked.py --dataset casia
    python scripts/train_chunked.py --dataset receipts
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gc

from src.config import get_config
from src.training import get_trainer
from src.utils import plot_training_curves, plot_chunked_training_progress, generate_training_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train forgery detection model")
    
    parser.add_argument('--dataset', type=str, default='doctamper',
                       choices=['doctamper', 'rtm', 'casia', 'receipts', 'fcd', 'scd'],
                       help='Dataset to train on')
    
    parser.add_argument('--chunk', type=int, default=None,
                       help='Chunk number (1-4) for DocTamper chunked training')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume from')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    return parser.parse_args()


def train_chunk(config, dataset_name: str, chunk_id: int, epochs: int = None, resume: str = None):
    """Train a single chunk"""
    
    # Calculate chunk boundaries
    chunks = config.get('data.chunked_training.chunks', [])
    
    if chunk_id > len(chunks):
        raise ValueError(f"Invalid chunk ID: {chunk_id}. Max: {len(chunks)}")
    
    chunk_config = chunks[chunk_id - 1]
    chunk_start = chunk_config['start']
    chunk_end = chunk_config['end']
    chunk_name = chunk_config['name']
    
    print(f"\n{'='*60}")
    print(f"Training Chunk {chunk_id}: {chunk_name}")
    print(f"Range: {chunk_start*100:.0f}% - {chunk_end*100:.0f}%")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = get_trainer(config, dataset_name)
    
    # Resume from previous chunk if applicable
    if resume:
        # For chunked training, reset epoch counter to train full epochs on new data
        trainer.load_checkpoint(resume, reset_epoch=True)
    elif chunk_id > 1:
        # Auto-resume from previous chunk
        prev_checkpoint = f'{dataset_name}_chunk{chunk_id-1}_final.pth'
        if (Path(config.get('outputs.checkpoints')) / prev_checkpoint).exists():
            print(f"Auto-resuming from previous chunk: {prev_checkpoint}")
            trainer.load_checkpoint(prev_checkpoint, reset_epoch=True)
    
    # Train
    history = trainer.train(
        epochs=epochs,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        chunk_id=chunk_id,
        resume_from=None  # Already loaded above
    )
    
    # Plot training curves
    plot_dir = Path(config.get('outputs.plots', 'outputs/plots'))
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plot_dir / f'{dataset_name}_chunk{chunk_id}_curves.png'
    plot_training_curves(
        history, 
        str(plot_path),
        title=f"{dataset_name.upper()} Chunk {chunk_id} Training"
    )
    
    # Generate report
    report_path = plot_dir / f'{dataset_name}_chunk{chunk_id}_report.txt'
    generate_training_report(history, str(report_path), f"{dataset_name} Chunk {chunk_id}")
    
    # Clear memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return history


def train_full_dataset(config, dataset_name: str, epochs: int = None, resume: str = None):
    """Train on full dataset (for smaller datasets)"""
    
    print(f"\n{'='*60}")
    print(f"Training on: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = get_trainer(config, dataset_name)
    
    # Load checkpoint if resuming (reset epoch counter for new dataset)
    if resume:
        print(f"Loading weights from: {resume}")
        trainer.load_checkpoint(resume, reset_epoch=True)
        print("Epoch counter reset to 0 for new dataset training")
    
    # Train
    history = trainer.train(
        epochs=epochs,
        chunk_id=0,
        resume_from=None  # Already loaded above
    )
    
    # Plot training curves
    plot_dir = Path(config.get('outputs.plots', 'outputs/plots'))
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plot_dir / f'{dataset_name}_training_curves.png'
    plot_training_curves(
        history,
        str(plot_path),
        title=f"{dataset_name.upper()} Training"
    )
    
    # Generate report
    report_path = plot_dir / f'{dataset_name}_report.txt'
    generate_training_report(history, str(report_path), dataset_name)
    
    return history


def main():
    args = parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("\n" + "="*60)
    print("Hybrid Document Forgery Detection - Training")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {config.get('system.device')}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60)
    
    # DocTamper: chunked training
    if args.dataset == 'doctamper' and args.chunk is not None:
        history = train_chunk(
            config, 
            args.dataset, 
            args.chunk,
            epochs=args.epochs,
            resume=args.resume
        )
    
    # DocTamper: all chunks sequentially
    elif args.dataset == 'doctamper' and args.chunk is None:
        print("Training DocTamper in 4 chunks...")
        
        all_histories = []
        for chunk_id in range(1, 5):
            history = train_chunk(
                config,
                args.dataset,
                chunk_id,
                epochs=args.epochs,
                resume=None if chunk_id == 1 else None  # Auto-resume from prev chunk
            )
            all_histories.append(history)
        
        # Plot combined progress
        plot_dir = Path(config.get('outputs.plots', 'outputs/plots'))
        combined_path = plot_dir / 'doctamper_all_chunks_progress.png'
        plot_chunked_training_progress(
            all_histories,
            str(combined_path),
            title="DocTamper Chunked Training Progress"
        )
    
    # Other datasets: full training
    else:
        history = train_full_dataset(
            config,
            args.dataset,
            epochs=args.epochs,
            resume=args.resume
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
