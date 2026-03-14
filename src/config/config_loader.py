"""
Configuration loader for Hybrid Document Forgery Detection System
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.encoder.name')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get dataset-specific configuration
        
        Args:
            dataset_name: Dataset name (doctamper, rtm, casia, receipts)
        
        Returns:
            Dataset configuration dictionary
        """
        return self.config['data']['datasets'].get(dataset_name, {})
    
    def has_pixel_mask(self, dataset_name: str) -> bool:
        """Check if dataset has pixel-level masks"""
        dataset_config = self.get_dataset_config(dataset_name)
        return dataset_config.get('has_pixel_mask', False)
    
    def should_skip_deskew(self, dataset_name: str) -> bool:
        """Check if deskewing should be skipped for dataset"""
        dataset_config = self.get_dataset_config(dataset_name)
        return dataset_config.get('skip_deskew', False)
    
    def should_skip_denoising(self, dataset_name: str) -> bool:
        """Check if denoising should be skipped for dataset"""
        dataset_config = self.get_dataset_config(dataset_name)
        return dataset_config.get('skip_denoising', False)
    
    def get_min_region_area(self, dataset_name: str) -> float:
        """Get minimum region area threshold for dataset"""
        dataset_config = self.get_dataset_config(dataset_name)
        return dataset_config.get('min_region_area', 0.001)
    
    def should_compute_localization_metrics(self, dataset_name: str) -> bool:
        """Check if localization metrics should be computed for dataset"""
        compute_config = self.config['metrics'].get('compute_localization', {})
        return compute_config.get(dataset_name, False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
_config = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
