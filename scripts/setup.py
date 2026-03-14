"""
Setup Script

Creates output directories and verifies installation.

Usage:
    python scripts/setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_directories():
    """Create required output directories"""
    
    base_dir = Path(__file__).parent.parent
    
    directories = [
        base_dir / 'outputs',
        base_dir / 'outputs' / 'checkpoints',
        base_dir / 'outputs' / 'logs',
        base_dir / 'outputs' / 'plots',
        base_dir / 'outputs' / 'results',
        base_dir / 'outputs' / 'classifier',
        base_dir / 'outputs' / 'exported',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")


def verify_installation():
    """Verify all required packages are installed"""
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('timm', 'TIMM'),
        ('lightgbm', 'LightGBM'),
        ('sklearn', 'Scikit-learn'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('albumentations', 'Albumentations'),
        ('tqdm', 'TQDM'),
        ('yaml', 'PyYAML'),
        ('pywt', 'PyWavelets'),
    ]
    
    print("\nVerifying installation...")
    print("-" * 40)
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (MISSING)")
            missing.append(name)
    
    # Check CUDA
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA Not Available (CPU mode)")
    except Exception as e:
        print(f"  ✗ CUDA Check Failed: {e}")
    
    return missing


def verify_datasets():
    """Verify dataset paths exist"""
    
    base_dir = Path(__file__).parent.parent
    
    datasets = {
        'DocTamper': base_dir / 'datasets' / 'DocTamper',
        'RTM': base_dir / 'datasets' / 'RealTextManipulation',
        'CASIA': base_dir / 'datasets' / 'CASIA 1.0 dataset',
        'Receipts': base_dir / 'datasets' / 'findit2',
    }
    
    print("\nVerifying datasets...")
    print("-" * 40)
    
    for name, path in datasets.items():
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND ({path})")


def main():
    print("\n" + "="*60)
    print("Hybrid Document Forgery Detection - Setup")
    print("="*60)
    
    # Create directories
    print("\nCreating directories...")
    print("-" * 40)
    create_directories()
    
    # Verify installation
    missing = verify_installation()
    
    # Verify datasets
    verify_datasets()
    
    # Summary
    print("\n" + "="*60)
    if missing:
        print("Setup complete with WARNINGS")
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
    else:
        print("Setup Complete! All checks passed.")
    print("="*60)


if __name__ == '__main__':
    main()
