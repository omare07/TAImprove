import numpy as np
import cv2
from pathlib import Path
import os
import glob
from typing import List, Tuple, Optional, Dict, Any, Union
import tifffile
from skimage import io, img_as_float, img_as_ubyte
import logging
import json
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class ImageLoader:
    """
    Utility class for loading histology images and tissue masks.
    
    Handles various image formats commonly used in histology and provides
    robust loading with error handling and format conversion.
    """
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """
        Load a histology image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as float array in range [0, 1]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Try different loading methods based on file extension
            extension = image_path.suffix.lower()
            
            if extension in ['.tif', '.tiff']:
                # Use tifffile for TIFF images (common in histology)
                image = tifffile.imread(str(image_path))
            else:
                # Use skimage for other formats
                image = io.imread(str(image_path))
            
            # Convert to float and normalize to [0, 1]
            image = img_as_float(image)
            
            # Ensure RGB format
            if image.ndim == 2:
                # Convert grayscale to RGB
                image = np.stack([image] * 3, axis=2)
            elif image.ndim == 3 and image.shape[2] == 4:
                # Remove alpha channel if present
                image = image[:, :, :3]
            elif image.ndim == 3 and image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    @staticmethod
    def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
        """
        Load a tissue mask from file.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Binary mask as boolean array
        """
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        try:
            # Load mask
            extension = mask_path.suffix.lower()
            
            if extension in ['.tif', '.tiff']:
                mask = tifffile.imread(str(mask_path))
            else:
                mask = io.imread(str(mask_path))
            
            # Convert to binary
            if mask.ndim == 3:
                # If RGB, use first channel or convert to grayscale
                mask = mask[:, :, 0] if mask.shape[2] > 1 else mask.squeeze()
            
            # Ensure binary (boolean) format
            if mask.dtype == bool:
                return mask
            else:
                # Threshold at 50% for conversion to binary
                return mask > (np.max(mask) * 0.5)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load mask {mask_path}: {str(e)}")
    
    @staticmethod
    def save_mask(mask: np.ndarray, output_path: Union[str, Path], 
                  format_type: str = 'tiff') -> None:
        """
        Save a tissue mask to file.
        
        Args:
            mask: Binary mask to save
            output_path: Output file path
            format_type: Output format ('tiff', 'png', etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type.lower() in ['tif', 'tiff']:
                # Save as TIFF using tifffile
                mask_uint8 = img_as_ubyte(mask.astype(float))
                tifffile.imwrite(str(output_path), mask_uint8)
            else:
                # Save using skimage
                mask_uint8 = img_as_ubyte(mask.astype(float))
                io.imsave(str(output_path), mask_uint8)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save mask to {output_path}: {str(e)}")


class FileManager:
    """
    Utility class for managing file operations and batch processing.
    
    Handles file discovery, pairing of images with masks, and batch operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the file manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def find_image_mask_pairs(self, image_dir: Union[str, Path], 
                            mask_dir: Union[str, Path],
                            image_extensions: List[str] = None,
                            mask_extensions: List[str] = None) -> List[Tuple[Path, Path]]:
        """
        Find matching image and mask file pairs.
        
        Args:
            image_dir: Directory containing original images
            mask_dir: Directory containing tissue masks
            image_extensions: List of valid image extensions
            mask_extensions: List of valid mask extensions
            
        Returns:
            List of (image_path, mask_path) tuples
        """
        if image_extensions is None:
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        if mask_extensions is None:
            mask_extensions = ['.tif', '.tiff', '.png']
        
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        # Find corresponding mask files
        pairs = []
        for image_path in image_files:
            # Look for matching mask file
            base_name = image_path.stem
            
            mask_found = False
            for ext in mask_extensions:
                potential_mask = mask_dir / f"{base_name}{ext}"
                if potential_mask.exists():
                    pairs.append((image_path, potential_mask))
                    mask_found = True
                    break
            
            if not mask_found:
                self.logger.warning(f"No matching mask found for image: {image_path}")
        
        self.logger.info(f"Found {len(pairs)} image-mask pairs")
        return pairs
    
    def create_output_structure(self, output_dir: Union[str, Path], 
                              subdirs: List[str] = None) -> Dict[str, Path]:
        """
        Create output directory structure.
        
        Args:
            output_dir: Base output directory
            subdirs: List of subdirectories to create
            
        Returns:
            Dictionary mapping subdirectory names to paths
        """
        if subdirs is None:
            subdirs = ['improved_masks', 'visualizations', 'reports', 'intermediate']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {'base': output_dir}
        
        for subdir in subdirs:
            subdir_path = output_dir / subdir
            subdir_path.mkdir(exist_ok=True)
            paths[subdir] = subdir_path
        
        return paths
    
    def save_processing_report(self, results: List[Dict[str, Any]], 
                             output_path: Union[str, Path]) -> None:
        """
        Save processing report as JSON.
        
        Args:
            results: List of processing results
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'summary': self._generate_summary_stats(results),
            'individual_results': results
        }
        
        # Convert numpy arrays and other non-serializable objects
        report = self._make_json_serializable(report)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Processing report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def _generate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {}
        
        # Extract metrics
        metrics = []
        processing_times = []
        artifact_counts = []
        
        for result in results:
            if 'improvement_metrics' in result:
                metrics.append(result['improvement_metrics'])
            if 'processing_time' in result:
                processing_times.append(result['processing_time'])
            if 'artifacts_detected' in result:
                artifact_counts.append(len(result['artifacts_detected']))
        
        summary = {}
        
        if metrics:
            # Calculate statistics for each metric
            metric_keys = metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    summary[f'{key}_mean'] = np.mean(values)
                    summary[f'{key}_std'] = np.std(values)
                    summary[f'{key}_median'] = np.median(values)
        
        if processing_times:
            summary['processing_time_mean'] = np.mean(processing_times)
            summary['processing_time_std'] = np.std(processing_times)
            summary['total_processing_time'] = np.sum(processing_times)
        
        if artifact_counts:
            summary['artifacts_per_image_mean'] = np.mean(artifact_counts)
            summary['total_artifacts_detected'] = np.sum(artifact_counts)
        
        return summary
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj


class ConfigManager:
    """
    Configuration management for tissue mask improvement.
    
    Handles loading, saving, and validation of configuration parameters.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._get_default_config()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'processing': {
                'min_tissue_intensity': 10,
                'max_background_intensity': 250,
                'saturation_threshold': 0.95,
                'close_kernel_size': 3,
                'open_kernel_size': 2,
                'min_object_area': 1000,
                'area_ratio_threshold': 0.02,
                'texture_window': 15,
                'bubble_circularity_threshold': 0.8,
                'glare_intensity_threshold': 240,
                'edge_strength_threshold': 0.1,
                'color_std_threshold': 5,
                'he_green_threshold': 210,
                'density_threshold': 0.05,
                'max_hole_area': 5000
            },
            'file_handling': {
                'image_extensions': ['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
                'mask_extensions': ['.tif', '.tiff', '.png'],
                'output_format': 'tiff',
                'save_intermediate': False
            },
            'visualization': {
                'create_comparison_plots': True,
                'create_artifact_plots': True,
                'create_batch_summary': True,
                'save_plots': True,
                'plot_dpi': 300
            },
            'logging': {
                'verbose': True,
                'log_level': 'INFO',
                'log_file': None
            }
        }
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Optional path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default config
            self._merge_config(self.config, loaded_config)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save configuration
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("No configuration path specified")
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing-specific configuration."""
        return self.config.get('processing', {})
    
    def get_file_config(self) -> Dict[str, Any]:
        """Get file handling configuration."""
        return self.config.get('file_handling', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def _merge_config(self, base_config: Dict[str, Any], 
                     new_config: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('TissueMaskImprovement')
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    level_str = config.get('log_level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if config.get('verbose', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    log_file = config.get('log_file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_paths(image_dir: Union[str, Path], mask_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Validate that input directories exist and are accessible.
    
    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing tissue masks
        
    Returns:
        Tuple of validated Path objects
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_path}")
    if not image_path.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_path}")
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_path}")
    if not mask_path.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {mask_path}")
    
    return image_path, mask_path 