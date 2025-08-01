import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import mode
from skimage import filters, morphology, segmentation, feature, measure
from skimage.color import rgb2gray
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class TissueMaskImprover:
    """
    Advanced tissue mask improvement system for histology images.
    
    This class implements sophisticated algorithms to refine binary tissue masks
    by analyzing original histology images and correcting common artifacts like
    air bubbles, glare streaks, and background noise.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tissue mask improver.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.logger = self._setup_logger()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            # Intensity thresholds
            'min_tissue_intensity': 10,
            'max_background_intensity': 250,
            'saturation_threshold': 0.95,
            
            # Morphological operations
            'close_kernel_size': 3,
            'open_kernel_size': 2,
            'min_object_area': 1000,
            'area_ratio_threshold': 0.02,
            
            # Texture analysis
            'glcm_distance': 1,
            'glcm_angles': [0, 45, 90, 135],
            'texture_window': 15,
            
            # Artifact detection
            'bubble_circularity_threshold': 0.8,
            'glare_intensity_threshold': 240,
            'edge_strength_threshold': 0.1,
            
            # Color analysis
            'color_std_threshold': 5,
            'he_green_threshold': 210,
            
            # Post-processing
            'density_threshold': 0.05,
            'max_hole_area': 5000,
            
            # Debugging
            'save_intermediate': False,
            'verbose': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('TissueMaskImprover')
        logger.setLevel(logging.INFO if self.config['verbose'] else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def improve_mask(self, original_image: np.ndarray, current_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Improve tissue mask using advanced image analysis.
        
        Args:
            original_image: Original histology image (H x W x 3)
            current_mask: Current binary tissue mask (H x W)
            
        Returns:
            Tuple of (improved_mask, metadata_dict)
        """
        self.logger.info("Starting tissue mask improvement...")
        
        # Input validation
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError("Original image must be RGB (H x W x 3)")
        if current_mask.ndim != 2:
            raise ValueError("Current mask must be 2D binary image")
        if original_image.shape[:2] != current_mask.shape:
            raise ValueError("Image and mask dimensions must match")
        
        metadata = {'steps': [], 'artifacts_detected': []}
        
        # Step 1: Analyze image properties
        image_props = self._analyze_image_properties(original_image)
        metadata['image_properties'] = image_props
        metadata['steps'].append('image_analysis')
        
        # Step 2: Detect and remove artifacts
        artifact_mask = self._detect_artifacts(original_image, image_props)
        metadata['artifacts_detected'] = self._categorize_artifacts(artifact_mask, original_image)
        metadata['steps'].append('artifact_detection')
        
        # Step 3: Enhanced tissue segmentation
        enhanced_mask = self._enhanced_tissue_segmentation(original_image, image_props)
        metadata['steps'].append('enhanced_segmentation')
        
        # Step 4: Combine information sources
        combined_mask = self._combine_masks(current_mask, enhanced_mask, artifact_mask)
        metadata['steps'].append('mask_combination')
        
        # Step 5: Morphological refinement
        refined_mask = self._morphological_refinement(combined_mask, original_image)
        metadata['steps'].append('morphological_refinement')
        
        # Step 6: Final quality control
        final_mask = self._quality_control(refined_mask, original_image)
        metadata['steps'].append('quality_control')
        
        # Calculate improvement metrics
        metadata['improvement_metrics'] = self._calculate_improvement_metrics(
            current_mask, final_mask, original_image
        )
        
        self.logger.info(f"Mask improvement completed. Steps: {len(metadata['steps'])}")
        return final_mask, metadata
    
    def _analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze basic image properties to guide processing."""
        props = {}
        
        # Convert to different color spaces
        gray = rgb2gray(image)
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Basic statistics
        props['mean_intensity'] = np.mean(image)
        props['std_intensity'] = np.std(image)
        props['mean_green'] = np.mean(image[:, :, 1])
        
        # Determine staining type (H&E vs IHC)
        props['is_he_stain'] = props['mean_green'] < self.config['he_green_threshold']
        props['stain_type'] = 'H&E' if props['is_he_stain'] else 'IHC'
        
        # Background estimation
        background_mask = (gray > 0.95) | (gray < 0.05)
        background_mask_bool = background_mask.astype(bool)
        if np.any(~background_mask_bool):
            props['background_color'] = np.median(image[~background_mask_bool], axis=0)
        else:
            props['background_color'] = np.array([1.0, 1.0, 1.0])
        
        # Saturation analysis
        saturation = hsv[:, :, 1] / 255.0
        props['mean_saturation'] = np.mean(saturation)
        props['high_saturation_ratio'] = np.mean(saturation > self.config['saturation_threshold'])
        
        self.logger.info(f"Detected {props['stain_type']} staining")
        return props
    
    def _detect_artifacts(self, image: np.ndarray, props: Dict[str, Any]) -> np.ndarray:
        """Detect various artifacts in the image."""
        h, w = image.shape[:2]
        artifact_mask = np.zeros((h, w), dtype=bool)
        
        gray = rgb2gray(image)
        
        # 1. Detect pure white/black regions (glare and holes)
        pure_white = np.all(image > 0.98, axis=2)
        pure_black = np.all(image < 0.02, axis=2)
        artifact_mask |= pure_white | pure_black
        
        # 2. Detect air bubbles (circular, low texture regions)
        bubble_mask = self._detect_air_bubbles(image, gray)
        artifact_mask |= bubble_mask
        
        # 3. Detect glare streaks (bright, elongated regions)
        glare_mask = self._detect_glare_streaks(image, gray)
        artifact_mask |= glare_mask
        
        # 4. Detect transparent regions (low saturation, uniform color)
        transparent_mask = self._detect_transparent_regions(image)
        artifact_mask |= transparent_mask
        
        # 5. Edge artifacts (image boundaries with sudden intensity changes)
        edge_artifacts = self._detect_edge_artifacts(gray)
        artifact_mask |= edge_artifacts
        
        # Morphological cleanup
        artifact_mask = morphology.binary_opening(artifact_mask, morphology.disk(2))
        
        return artifact_mask
    
    def _detect_air_bubbles(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Detect air bubbles - typically circular regions with uniform intensity."""
        # Find regions with low texture
        texture = filters.rank.entropy(gray, morphology.disk(5))
        low_texture = texture < np.percentile(texture, 10)
        
        # Find circular regions
        labeled = measure.label(low_texture)
        bubble_mask = np.zeros_like(gray, dtype=bool)
        
        for region in measure.regionprops(labeled):
            if region.area > 100:  # Minimum size threshold
                # Calculate circularity
                perimeter = region.perimeter
                area = region.area
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity > self.config['bubble_circularity_threshold']:
                    # Check if region has uniform intensity
                    mask = labeled == region.label
                    region_std = np.std(gray[mask])
                    if region_std < 0.05:  # Low intensity variation
                        bubble_mask |= mask
        
        return bubble_mask
    
    def _detect_glare_streaks(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Detect glare streaks - bright, elongated regions."""
        # High intensity regions
        bright_mask = gray > (self.config['glare_intensity_threshold'] / 255.0)
        
        # Remove small isolated pixels
        bright_mask = morphology.binary_opening(bright_mask, morphology.disk(3))
        
        # Find elongated regions
        labeled = measure.label(bright_mask)
        glare_mask = np.zeros_like(gray, dtype=bool)
        
        for region in measure.regionprops(labeled):
            if region.area > 200:  # Minimum size
                # Calculate aspect ratio
                minor_axis = region.minor_axis_length
                major_axis = region.major_axis_length
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
                
                if aspect_ratio > 3:  # Elongated shape
                    glare_mask |= labeled == region.label
        
        return glare_mask
    
    def _detect_transparent_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect transparent or poorly stained regions."""
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        
        # Low saturation regions
        low_sat = saturation < 0.1
        
        # Regions with very uniform color
        color_std = np.std(image, axis=2)
        uniform_color = color_std < (self.config['color_std_threshold'] / 255.0)
        
        return low_sat & uniform_color
    
    def _detect_edge_artifacts(self, gray: np.ndarray) -> np.ndarray:
        """Detect artifacts at image edges."""
        h, w = gray.shape
        edge_mask = np.zeros((h, w), dtype=bool)
        
        # Check border regions for sudden intensity changes
        border_width = 10
        
        # Top and bottom edges
        top_mean = np.mean(gray[:border_width, :])
        bottom_mean = np.mean(gray[-border_width:, :])
        
        # Left and right edges  
        left_mean = np.mean(gray[:, :border_width])
        right_mean = np.mean(gray[:, -border_width:])
        
        overall_mean = np.mean(gray)
        
        # Mark edge regions if they're significantly different from overall image
        if abs(top_mean - overall_mean) > 0.3:
            edge_mask[:border_width, :] = True
        if abs(bottom_mean - overall_mean) > 0.3:
            edge_mask[-border_width:, :] = True
        if abs(left_mean - overall_mean) > 0.3:
            edge_mask[:, :border_width] = True
        if abs(right_mean - overall_mean) > 0.3:
            edge_mask[:, -border_width:] = True
        
        return edge_mask
    
    def _enhanced_tissue_segmentation(self, image: np.ndarray, props: Dict[str, Any]) -> np.ndarray:
        """Perform enhanced tissue segmentation using multiple features."""
        gray = rgb2gray(image)
        
        if props['is_he_stain']:
            mask = self._segment_he_tissue(image, gray, props)
        else:
            mask = self._segment_ihc_tissue(image, gray, props)
        
        return mask
    
    def _segment_he_tissue(self, image: np.ndarray, gray: np.ndarray, props: Dict[str, Any]) -> np.ndarray:
        """Segment tissue in H&E stained images."""
        # Multi-feature approach for H&E
        
        # 1. Intensity-based segmentation
        background_diff = np.abs(image - props['background_color'])
        intensity_mask = np.mean(background_diff, axis=2) > (self.config['min_tissue_intensity'] / 255.0)
        
        # 2. Texture-based segmentation
        texture = filters.rank.entropy(gray, morphology.disk(self.config['texture_window'] // 2))
        texture_mask = texture > np.percentile(texture, 30)
        
        # 3. Color variance segmentation (remove regions with low color variation)
        color_std = np.std(image, axis=2)
        variance_mask = color_std > (self.config['color_std_threshold'] / 255.0)
        
        # 4. Edge-based segmentation
        edges = feature.canny(gray, sigma=1, low_threshold=0.1, high_threshold=0.2)
        edge_density = ndimage.uniform_filter(edges.astype(float), size=15)
        edge_mask = edge_density > self.config['edge_strength_threshold']
        
        # Combine all features
        combined = intensity_mask & texture_mask & variance_mask & edge_mask
        
        return combined
    
    def _segment_ihc_tissue(self, image: np.ndarray, gray: np.ndarray, props: Dict[str, Any]) -> np.ndarray:
        """Segment tissue in IHC stained images."""
        # IHC images often have different characteristics
        
        # 1. Deviation from background
        background_diff = np.abs(np.mean(image, axis=2) - np.mean(props['background_color']))
        intensity_mask = background_diff > (self.config['min_tissue_intensity'] / 255.0)
        
        # 2. Morphological approach
        mask = morphology.binary_closing(intensity_mask, morphology.disk(self.config['close_kernel_size']))
        mask = ndimage.binary_fill_holes(mask)
        
        # 3. Keep only largest connected component for IHC
        labeled = measure.label(mask)
        if labeled.max() > 0:
            largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            mask = labeled == largest_label
        
        return mask
    
    def _combine_masks(self, current_mask: np.ndarray, enhanced_mask: np.ndarray, 
                      artifact_mask: np.ndarray) -> np.ndarray:
        """Combine information from different mask sources."""
        # Start with enhanced segmentation
        combined = enhanced_mask.copy()
        
        # Add regions from current mask that aren't artifacts
        # Ensure masks are boolean before invert operation
        artifact_mask_bool = artifact_mask.astype(bool)
        valid_current = current_mask & ~artifact_mask_bool
        combined |= valid_current
        
        # Aggressively remove ALL detected artifacts to ensure complete elimination
        combined = self._smooth_artifact_removal_generic(combined, artifact_mask)
        
        return combined
    
    def _morphological_refinement(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply conservative morphological operations to refine the mask without creating patterns."""
        # Use conservative kernel sizes to prevent checkerboard patterns
        conservative_open_size = min(self.config['open_kernel_size'], 2)
        conservative_close_size = min(self.config['close_kernel_size'], 3)
        
        # Conservative opening to remove small noise without creating sharp edges
        refined = morphology.binary_opening(mask, morphology.disk(conservative_open_size))
        
        # Conservative closing to fill small gaps without over-connecting
        refined = morphology.binary_closing(refined, morphology.disk(conservative_close_size))
        
        # Remove small objects (but not too aggressively)
        min_area = min(self.config['min_object_area'], 2000)  # Cap at reasonable size
        refined = morphology.remove_small_objects(refined, min_size=min_area)
        
        # Fill holes conservatively - only small holes to preserve tissue structure
        # Ensure refined is boolean before invert operation to prevent type errors
        refined_bool = refined.astype(bool)
        labeled_holes = measure.label(~refined_bool)
        max_hole_size = min(self.config['max_hole_area'], 1000)  # Conservative hole filling
        for region in measure.regionprops(labeled_holes):
            if region.area < max_hole_size:
                refined[labeled_holes == region.label] = True
        
        return refined
    
    def _quality_control(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Perform final quality control on the mask."""
        # Check tissue density
        tissue_density = np.mean(mask)
        
        if tissue_density < self.config['density_threshold']:
            self.logger.warning(f"Low tissue density detected: {tissue_density:.3f}")
            # Apply fallback strategy for low density images
            mask = self._fallback_segmentation(image)
        
        # Remove objects that are too small relative to the largest object
        labeled = measure.label(mask)
        if labeled.max() > 0:
            sizes = np.bincount(labeled.flat)
            sizes[0] = 0  # Ignore background
            max_size = np.max(sizes)
            min_allowed = max_size * self.config['area_ratio_threshold']
            
            for i in range(1, len(sizes)):
                if sizes[i] < min_allowed:
                    mask[labeled == i] = False
        
        return mask
    
    def _fallback_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Fallback segmentation method for difficult images."""
        gray = rgb2gray(image)
        
        # Simple threshold-based approach
        threshold = filters.threshold_otsu(gray)
        mask = gray < threshold
        
        # Morphological cleanup
        mask = morphology.binary_closing(mask, morphology.disk(4))
        mask = morphology.remove_small_objects(mask, min_size=self.config['min_object_area'])
        
        return mask
    
    def _categorize_artifacts(self, artifact_mask: np.ndarray, image: np.ndarray) -> list:
        """Categorize detected artifacts for reporting."""
        artifacts = []
        labeled = measure.label(artifact_mask)
        
        for region in measure.regionprops(labeled):
            artifact_info = {
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox
            }
            
            # Simple categorization based on properties
            if region.area > 10000:
                artifact_info['type'] = 'large_artifact'
            elif region.eccentricity > 0.8:
                artifact_info['type'] = 'linear_artifact'
            else:
                artifact_info['type'] = 'spot_artifact'
            
            artifacts.append(artifact_info)
        
        return artifacts
    
    def _calculate_improvement_metrics(self, original_mask: np.ndarray, 
                                     improved_mask: np.ndarray, image: np.ndarray) -> Dict[str, float]:
        """Calculate metrics to quantify mask improvement."""
        metrics = {}
        
        # Basic area comparison
        original_area = np.sum(original_mask)
        improved_area = np.sum(improved_mask)
        metrics['area_change_ratio'] = improved_area / original_area if original_area > 0 else 0
        
        # Overlap metrics
        intersection = np.sum(original_mask & improved_mask)
        union = np.sum(original_mask | improved_mask)
        metrics['jaccard_index'] = intersection / union if union > 0 else 0
        
        # Dice coefficient
        metrics['dice_coefficient'] = 2 * intersection / (original_area + improved_area) if (original_area + improved_area) > 0 else 0
        
        # Artifact reduction (approximate)
        gray = rgb2gray(image)
        # High intensity regions in tissue mask (potential artifacts)
        bright_in_original = np.sum((gray > 0.9) & original_mask)
        bright_in_improved = np.sum((gray > 0.9) & improved_mask)
        metrics['bright_artifact_reduction'] = (bright_in_original - bright_in_improved) / bright_in_original if bright_in_original > 0 else 0
        
        return metrics
    
    def _smooth_artifact_removal_generic(self, tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """
        Aggressively but smoothly remove ALL detected artifacts to ensure complete elimination.
        
        This method ensures artifacts are fully removed while maintaining smooth boundaries 
        to prevent checkerboard patterns.
        """
        if not np.any(artifact_mask):
            return tissue_mask
        
        result = tissue_mask.copy()
        artifact_mask_bool = artifact_mask.astype(bool)
        
        # Method 1: AGGRESSIVE dilation of artifact mask to catch halos and borders
        # This ensures we remove artifact boundaries and surrounding areas
        dilated_artifacts = morphology.binary_dilation(artifact_mask_bool, morphology.disk(4))
        
        # Method 2: Apply morphological reconstruction with larger safety margin
        # Erode less aggressively to preserve more of the artifact detection
        artifact_markers = morphology.binary_erosion(artifact_mask_bool, morphology.disk(1))
        if np.any(artifact_markers):
            reconstructed_artifacts = morphology.reconstruction(artifact_markers, dilated_artifacts)
            reconstructed_artifacts_bool = reconstructed_artifacts.astype(bool)
        else:
            reconstructed_artifacts_bool = dilated_artifacts
        
        # Method 3: Distance-based removal with AGGRESSIVE safety zone
        from scipy.ndimage import distance_transform_edt
        
        # Create larger safety zones around ALL artifacts
        artifact_distance = distance_transform_edt(~artifact_mask_bool)
        
        # AGGRESSIVE removal threshold - remove everything within 6 pixels of any artifact
        aggressive_removal_threshold = 6.0  # pixels
        aggressive_removal_zone = artifact_distance < aggressive_removal_threshold
        
        # Method 4: Add color-based removal for potential air bubbles in generic case
        if hasattr(self, '_detect_color_based_artifacts'):
            color_artifacts = self._detect_color_based_artifacts(result, tissue_mask)
            aggressive_removal_zone = aggressive_removal_zone | color_artifacts
        
        # Combine all removal methods for complete elimination
        complete_removal_mask = reconstructed_artifacts_bool | aggressive_removal_zone.astype(bool)
        
        # Apply complete removal
        result = result & ~complete_removal_mask
        
        # Method 5: Final cleanup with minimal morphological operations
        # Only light cleanup to preserve the aggressive removal
        result = morphology.binary_opening(result, morphology.disk(1))
        
        # Log removal statistics
        artifacts_removed = np.sum(complete_removal_mask & tissue_mask)
        total_artifacts = np.sum(artifact_mask_bool)
        self.logger.info(f"Aggressively removed {artifacts_removed} artifact pixels (original: {total_artifacts})")
        
        return result
        
    def _detect_color_based_artifacts(self, current_mask: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """
        Additional color-based artifact detection for the generic improver.
        
        Detects very bright, low-saturation regions that might be air bubbles
        missed by the main detection.
        """
        # This is a simplified version for the generic case
        # In practice, this would use the original image, but we work with what we have
        
        # For now, return empty mask - this method is mainly a placeholder
        # for future enhancement
        return np.zeros_like(current_mask, dtype=bool) 