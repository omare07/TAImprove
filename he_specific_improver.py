import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure, color, feature
from skimage.color import rgb2gray, rgb2hsv
import logging
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class HESpecificImprover:
    """
    H&E-specific tissue mask improvement that leverages the predictable color palette.
    
    H&E Staining Characteristics:
    - Hematoxylin: Stains nuclei blue/purple (high blue channel)
    - Eosin: Stains cytoplasm pink/red (high red channel)
    - Air bubbles: Bright white/clear with no staining
    - Background: White/very light
    - True tissue: Contains visible staining (not pure white)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the H&E-specific improver."""
        self.config = self._get_he_config()
        if config:
            # Use he_specific config section if available
            if 'he_specific' in config:
                self.config.update(config['he_specific'])
            else:
                self.config.update(config)
        
        self.logger = logging.getLogger('HESpecificImprover')
        self.logger.setLevel(logging.INFO)
    
    def _get_he_config(self) -> Dict[str, Any]:
        """Get H&E-specific configuration parameters."""
        return {
            # H&E Color thresholds (normalized 0-1) - Made more inclusive
            'min_staining_intensity': 0.05,  # Lower threshold for darker staining
            'max_background_intensity': 0.85,  # Background/air bubble threshold
            
            # Hematoxylin (blue/purple) detection - More inclusive
            'hematoxylin_blue_min': 0.25,  # Lower threshold for darker blue staining
            'hematoxylin_saturation_min': 0.1,  # Lower saturation for subtle staining
            
            # Eosin (pink/red) detection - More inclusive
            'eosin_red_min': 0.2,  # Lower threshold for darker red staining
            'eosin_pink_ratio': 1.1,  # More lenient red/blue ratio
            
            # Air bubble detection
            'bubble_brightness_threshold': 0.9,  # Very bright regions
            'bubble_saturation_max': 0.1,  # Low saturation (nearly white)
            'bubble_min_circularity': 0.7,  # Circular shape
            'bubble_min_size': 100,  # Minimum bubble size
            
            # Morphological operations
            'close_kernel_size': 3,
            'open_kernel_size': 2,
            'min_tissue_area': 1000,
            
            # Quality control
            'max_tissue_brightness': 0.8,  # True tissue shouldn't be too bright
            
            # Linear artifact detection
            'streak_brightness_threshold': 0.8,  # Bright streaks
            'line_length_threshold': 100,  # Minimum line length to be considered artifact
            'line_aspect_ratio_min': 5,  # Minimum aspect ratio for linear objects
            'hough_threshold': 50,  # Hough transform threshold for line detection
        }
    
    def improve_he_mask(self, original_image: np.ndarray, current_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Improve tissue mask specifically for H&E stained images.
        
        Args:
            original_image: Original H&E histology image (H x W x 3)
            current_mask: Current binary tissue mask (H x W)
            
        Returns:
            Tuple of (improved_mask, metadata_dict)
        """
        self.logger.info("Starting H&E-specific mask improvement...")
        
        metadata = {'method': 'HE_specific', 'steps': [], 'artifacts_detected': []}
        
        # Step 1: Analyze H&E color characteristics
        he_analysis = self._analyze_he_staining(original_image)
        metadata['he_analysis'] = he_analysis
        metadata['steps'].append('he_color_analysis')
        
        # Step 2: Detect air bubbles and artifacts
        bubble_mask = self._detect_he_air_bubbles(original_image)
        artifact_count = np.sum(bubble_mask)
        metadata['artifacts_detected'] = [{'type': 'air_bubbles', 'count': int(artifact_count)}]
        metadata['steps'].append('bubble_detection')
        
        # Step 3: Create staining-based tissue mask
        staining_mask = self._create_he_staining_mask(original_image)
        metadata['steps'].append('staining_segmentation')
        
        # Step 4: Remove bright regions (likely artifacts)
        bright_regions = self._detect_bright_artifacts(original_image)
        metadata['steps'].append('bright_artifact_removal')
        
        # Step 5: Detect linear artifacts (streaks/lines)
        linear_artifacts = self._detect_linear_artifacts(original_image)
        linear_count = np.sum(linear_artifacts)
        metadata['artifacts_detected'].append({'type': 'linear_streaks', 'count': int(linear_count)})
        metadata['steps'].append('linear_artifact_detection')
        
        # Step 6: Combine information
        # Start with staining mask, remove bubbles, bright regions, and linear artifacts
        improved_mask = staining_mask & ~bubble_mask & ~bright_regions & ~linear_artifacts
        metadata['steps'].append('mask_combination')
        
        # Step 7: Morphological refinement
        improved_mask = self._morphological_cleanup(improved_mask)
        metadata['steps'].append('morphological_cleanup')
        
        # Step 8: Size filtering
        improved_mask = self._size_filtering(improved_mask)
        metadata['steps'].append('size_filtering')
        
        # Calculate improvement metrics
        metadata['improvement_metrics'] = self._calculate_he_metrics(
            current_mask, improved_mask, original_image
        )
        
        self.logger.info(f"H&E mask improvement completed. Removed {artifact_count} bubble pixels and {linear_count} linear artifact pixels.")
        return improved_mask, metadata
    
    def _analyze_he_staining(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze H&E staining characteristics in the image - more comprehensive detection."""
        # Convert to different color spaces
        hsv_image = rgb2hsv(image)
        gray = rgb2gray(image)
        
        # Analyze color channels
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1] 
        blue_channel = image[:, :, 2]
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        
        # More comprehensive hematoxylin detection (blue/purple nuclei)
        hematoxylin_rgb = (blue_channel > self.config['hematoxylin_blue_min']) & \
                         (saturation > self.config['hematoxylin_saturation_min'])
        
        # HSV-based hematoxylin (blue to purple spectrum)
        hematoxylin_hsv = ((hue >= 0.5) & (hue <= 0.83)) & (saturation > 0.1)
        hematoxylin_mask = hematoxylin_rgb | hematoxylin_hsv
        
        # More comprehensive eosin detection (pink/red cytoplasm)
        eosin_rgb = (red_channel > self.config['eosin_red_min']) & \
                   (red_channel > blue_channel * 0.8)
        
        # HSV-based eosin (pink to red spectrum)
        eosin_hsv = (((hue >= 0.83) & (hue <= 1.0)) | ((hue >= 0.0) & (hue <= 0.08))) & \
                   (saturation > 0.1)
        eosin_mask = eosin_rgb | eosin_hsv
        
        # Overall staining including darker regions
        stained_regions = hematoxylin_mask | eosin_mask
        
        # Include any regions with color variation (captures darker staining)
        color_variation = np.std(image, axis=2)
        has_color = color_variation > 0.02
        comprehensive_staining = stained_regions | has_color
        
        # Exclude pure white background
        tissue_regions = comprehensive_staining & (gray < 0.85)
        
        analysis = {
            'hematoxylin_percentage': float(np.mean(hematoxylin_mask) * 100),
            'eosin_percentage': float(np.mean(eosin_mask) * 100),
            'total_stained_percentage': float(np.mean(stained_regions) * 100),
            'comprehensive_tissue_percentage': float(np.mean(tissue_regions) * 100),
            'mean_red': float(np.mean(red_channel)),
            'mean_green': float(np.mean(green_channel)),
            'mean_blue': float(np.mean(blue_channel)),
            'red_blue_ratio': float(np.mean(red_channel) / (np.mean(blue_channel) + 0.001)),
            'color_variation_mean': float(np.mean(color_variation))
        }
        
        self.logger.info(f"H&E Analysis: {analysis['hematoxylin_percentage']:.1f}% hematoxylin, "
                        f"{analysis['eosin_percentage']:.1f}% eosin, "
                        f"{analysis['comprehensive_tissue_percentage']:.1f}% total tissue")
        
        return analysis
    
    def _detect_he_air_bubbles(self, image: np.ndarray) -> np.ndarray:
        """Detect air bubbles in H&E images using brightness and shape."""
        gray = rgb2gray(image)
        hsv_image = rgb2hsv(image)
        
        # Air bubbles are very bright and have low saturation (nearly white)
        bright_mask = gray > self.config['bubble_brightness_threshold']
        low_saturation = hsv_image[:, :, 1] < self.config['bubble_saturation_max']
        
        # Combine brightness and saturation criteria
        potential_bubbles = bright_mask & low_saturation
        
        # Remove small noise
        potential_bubbles = morphology.binary_opening(potential_bubbles, morphology.disk(2))
        
        # Analyze connected components for circularity
        labeled_bubbles = measure.label(potential_bubbles)
        bubble_mask = np.zeros_like(gray, dtype=bool)
        
        bubble_count = 0
        for region in measure.regionprops(labeled_bubbles):
            if region.area >= self.config['bubble_min_size']:
                # Calculate circularity (4π * area / perimeter²)
                perimeter = region.perimeter
                area = region.area
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity >= self.config['bubble_min_circularity']:
                    # This is likely an air bubble
                    bubble_mask[labeled_bubbles == region.label] = True
                    bubble_count += 1
        
        self.logger.info(f"Detected {bubble_count} air bubbles")
        return bubble_mask
    
    def _create_he_staining_mask(self, image: np.ndarray) -> np.ndarray:
        """Create tissue mask based on H&E staining patterns - more inclusive for all H&E hues."""
        # Convert to different color spaces
        gray = rgb2gray(image)
        hsv_image = rgb2hsv(image)
        
        # Basic intensity filtering - exclude pure white/bright regions
        not_background = gray < self.config['max_tissue_brightness']
        not_too_dark = gray > self.config['min_staining_intensity']
        intensity_mask = not_background & not_too_dark
        
        # HSV-based H&E detection (more comprehensive)
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]
        
        # Hematoxylin hues (blue/purple spectrum: 180-300 degrees in HSV)
        # Convert to 0-1 range: 0.5-0.83 (blue to purple)
        hematoxylin_hue_mask = ((hue >= 0.5) & (hue <= 0.83)) | \
                              ((hue >= 0.6) & (hue <= 1.0))  # Include violet/purple range
        hematoxylin_mask = hematoxylin_hue_mask & (saturation > self.config['hematoxylin_saturation_min'])
        
        # Eosin hues (pink/red spectrum: 300-360 and 0-30 degrees)
        # Convert to 0-1 range: 0.83-1.0 and 0.0-0.08 (pink to red)
        eosin_hue_mask = ((hue >= 0.83) & (hue <= 1.0)) | \
                        ((hue >= 0.0) & (hue <= 0.08))
        eosin_mask = eosin_hue_mask & (saturation > 0.1)  # Any saturation for eosin
        
        # RGB-based detection (backup for edge cases)
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
        
        # Enhanced blue detection (hematoxylin)
        blue_dominant = (blue_channel > self.config['hematoxylin_blue_min']) & \
                       (blue_channel > red_channel * 0.9) & \
                       (blue_channel > green_channel * 0.9)
        
        # Enhanced red/pink detection (eosin)
        red_dominant = (red_channel > self.config['eosin_red_min']) & \
                      (red_channel > blue_channel * 0.8)
        
        # Combine all staining detection methods
        has_hematoxylin = hematoxylin_mask | blue_dominant
        has_eosin = eosin_mask | red_dominant
        has_any_staining = has_hematoxylin | has_eosin
        
        # Also include regions with ANY visible color (not pure grayscale)
        color_variation = np.std(image, axis=2)
        has_color = color_variation > 0.02  # Any color variation indicates staining
        
        # Final mask: intensity OK AND (specific staining OR general color)
        staining_mask = intensity_mask & (has_any_staining | has_color)
        
        return staining_mask
    
    def _detect_bright_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Detect overly bright regions that are likely artifacts."""
        gray = rgb2gray(image)
        
        # Very bright regions
        bright_artifacts = gray > self.config['max_tissue_brightness']
        
        # Also check for regions with very little color variation (uniform bright areas)
        color_std = np.std(image, axis=2)
        uniform_bright = (gray > 0.8) & (color_std < 0.05)
        
        return bright_artifacts | uniform_bright
    
    def _detect_linear_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Detect linear artifacts like streaks, lines, and scanning artifacts."""
        gray = rgb2gray(image)
        h, w = gray.shape
        linear_mask = np.zeros((h, w), dtype=bool)
        
        # 1. Detect bright streaks (glare artifacts)
        bright_streaks = self._detect_bright_streaks(gray)
        linear_mask |= bright_streaks
        
        # 2. Detect elongated objects (linear structures)
        elongated_objects = self._detect_elongated_objects(gray)
        linear_mask |= elongated_objects
        
        # 3. Detect lines using Hough transform
        hough_lines = self._detect_hough_lines(gray)
        linear_mask |= hough_lines
        
        # 4. Detect horizontal and vertical artifacts (common in scanning)
        scanning_artifacts = self._detect_scanning_artifacts(gray)
        linear_mask |= scanning_artifacts
        
        return linear_mask
    
    def _detect_bright_streaks(self, gray: np.ndarray) -> np.ndarray:
        """Detect bright linear streaks (glare artifacts)."""
        # Find very bright regions
        bright_mask = gray > self.config['streak_brightness_threshold']
        
        # Use morphological operations to identify linear structures
        # Horizontal streaks
        horizontal_kernel = morphology.rectangle(1, 15)
        horizontal_streaks = morphology.opening(bright_mask, horizontal_kernel)
        
        # Vertical streaks  
        vertical_kernel = morphology.rectangle(15, 1)
        vertical_streaks = morphology.opening(bright_mask, vertical_kernel)
        
        # Diagonal streaks (45 degrees)
        diagonal_kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool)
        diagonal_streaks = morphology.opening(bright_mask, diagonal_kernel)
        
        return horizontal_streaks | vertical_streaks | diagonal_streaks
    
    def _detect_elongated_objects(self, gray: np.ndarray) -> np.ndarray:
        """Detect elongated objects that might be linear artifacts."""
        # Use edge detection to find boundaries
        edges = feature.canny(gray, sigma=1, low_threshold=0.1, high_threshold=0.2)
        
        # Find connected components
        labeled_edges = measure.label(edges)
        linear_objects = np.zeros_like(gray, dtype=bool)
        
        for region in measure.regionprops(labeled_edges):
            if region.area > 50:  # Minimum size
                # Calculate aspect ratio
                minor_axis = region.minor_axis_length
                major_axis = region.major_axis_length
                
                if minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    
                    # High aspect ratio indicates linear object
                    if aspect_ratio > self.config['line_aspect_ratio_min']:
                        linear_objects[labeled_edges == region.label] = True
        
        return linear_objects
    
    def _detect_hough_lines(self, gray: np.ndarray) -> np.ndarray:
        """Detect lines using Hough transform."""
        # Convert to uint8 for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Edge detection
        edges = cv2.Canny(gray_uint8, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.config['hough_threshold'],
                               minLineLength=self.config['line_length_threshold'], maxLineGap=10)
        
        line_mask = np.zeros_like(gray, dtype=bool)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw line on mask with some thickness
                cv2.line(line_mask.astype(np.uint8), (x1, y1), (x2, y2), 1, thickness=3)
        
        return line_mask.astype(bool)
    
    def _detect_scanning_artifacts(self, gray: np.ndarray) -> np.ndarray:
        """Detect horizontal/vertical scanning artifacts."""
        h, w = gray.shape
        artifacts = np.zeros((h, w), dtype=bool)
        
        # Check for horizontal lines (entire rows with similar intensity)
        for i in range(h):
            row = gray[i, :]
            row_std = np.std(row)
            row_mean = np.mean(row)
            
            # If row has very low variation and is bright, it might be an artifact
            if row_std < 0.02 and row_mean > 0.7:
                artifacts[i, :] = True
        
        # Check for vertical lines (entire columns with similar intensity)
        for j in range(w):
            col = gray[:, j]
            col_std = np.std(col)
            col_mean = np.mean(col)
            
            # If column has very low variation and is bright, it might be an artifact
            if col_std < 0.02 and col_mean > 0.7:
                artifacts[:, j] = True
        
        return artifacts
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the mask."""
        # Remove small noise
        cleaned = morphology.binary_opening(mask, morphology.disk(self.config['open_kernel_size']))
        
        # Fill small gaps
        cleaned = morphology.binary_closing(cleaned, morphology.disk(self.config['close_kernel_size']))
        
        # Fill holes in tissue regions
        cleaned = ndimage.binary_fill_holes(cleaned)
        
        return cleaned
    
    def _size_filtering(self, mask: np.ndarray) -> np.ndarray:
        """Remove objects that are too small to be meaningful tissue."""
        return morphology.remove_small_objects(mask, min_size=self.config['min_tissue_area'])
    
    def _calculate_he_metrics(self, original_mask: np.ndarray, 
                             improved_mask: np.ndarray, 
                             image: np.ndarray) -> Dict[str, float]:
        """Calculate H&E-specific improvement metrics."""
        metrics = {}
        
        # Basic area comparison
        original_area = np.sum(original_mask)
        improved_area = np.sum(improved_mask)
        metrics['area_change_ratio'] = improved_area / original_area if original_area > 0 else 0
        
        # Overlap metrics
        intersection = np.sum(original_mask & improved_mask)
        union = np.sum(original_mask | improved_mask)
        metrics['jaccard_index'] = intersection / union if union > 0 else 0
        
        # Bright region reduction (air bubble removal effectiveness)
        gray = rgb2gray(image)
        bright_in_original = np.sum((gray > 0.9) & original_mask)
        bright_in_improved = np.sum((gray > 0.9) & improved_mask)
        metrics['bright_region_reduction'] = (bright_in_original - bright_in_improved) / bright_in_original if bright_in_original > 0 else 0
        
        # Staining preservation (how much actual stained tissue is kept)
        stained_regions = (gray < 0.8) & (gray > 0.2)  # Regions with visible staining
        stained_in_original = np.sum(stained_regions & original_mask)
        stained_in_improved = np.sum(stained_regions & improved_mask)
        metrics['staining_preservation'] = stained_in_improved / stained_in_original if stained_in_original > 0 else 0
        
        return metrics


# Integration function to use this with the main system
def improve_he_mask_integration(original_image: np.ndarray, current_mask: np.ndarray, 
                               config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Integration function for H&E-specific mask improvement.
    
    This can be used as a drop-in replacement for the generic improver.
    """
    improver = HESpecificImprover(config)
    return improver.improve_he_mask(original_image, current_mask) 