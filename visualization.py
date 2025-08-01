import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, List, Optional, Tuple
import cv2
from skimage.color import rgb2gray
from skimage import measure
import warnings

warnings.filterwarnings('ignore')

class TissueMaskVisualizer:
    """
    Visualization tools for tissue mask improvement analysis.
    
    Provides comprehensive visualization capabilities for comparing original
    and improved tissue masks, analyzing artifacts, and generating reports.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        
    def create_comparison_plot(self, original_image: np.ndarray, 
                             original_mask: np.ndarray, 
                             improved_mask: np.ndarray,
                             metadata: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive comparison plot showing before/after results.
        
        Args:
            original_image: Original histology image
            original_mask: Original tissue mask
            improved_mask: Improved tissue mask
            metadata: Processing metadata
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure object
        """
        try:
            # Validate inputs
            if original_image is None or original_mask is None or improved_mask is None:
                raise ValueError("One or more input arrays is None")
            
            # Ensure all arrays have compatible shapes
            if original_image.shape[:2] != original_mask.shape[:2] or original_image.shape[:2] != improved_mask.shape[:2]:
                raise ValueError(f"Shape mismatch: image {original_image.shape[:2]}, "
                               f"original_mask {original_mask.shape[:2]}, "
                               f"improved_mask {improved_mask.shape[:2]}")
            
            # Normalize image to [0,1] if needed
            if original_image.max() > 1.0:
                original_image = original_image / 255.0
                
            # Ensure masks are binary
            original_mask = original_mask.astype(bool)
            improved_mask = improved_mask.astype(bool)
            
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # Row 1: Original data
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_image)
            ax1.set_title('Original Image', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(original_mask, cmap='gray')
            ax2.set_title('Original Mask', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(improved_mask, cmap='gray')
            ax3.set_title('Improved Mask', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[0, 3])
            overlay = self._create_mask_overlay(original_image, improved_mask)
            ax4.imshow(overlay)
            ax4.set_title('Improved Mask Overlay', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            # Row 2: Analysis
            ax5 = fig.add_subplot(gs[1, 0])
            difference = self._compute_mask_difference(original_mask, improved_mask)
            im5 = ax5.imshow(difference, cmap='RdYlBu', vmin=-1, vmax=1)
            ax5.set_title('Mask Difference\n(Blue: Added, Red: Removed)', fontsize=12, fontweight='bold')
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            
            # Artifacts if available
            ax6 = fig.add_subplot(gs[1, 1])
            if 'artifacts_detected' in metadata and len(metadata['artifacts_detected']) > 0:
                artifact_viz = self._visualize_artifacts(original_image, metadata['artifacts_detected'])
                ax6.imshow(artifact_viz)
                ax6.set_title(f'Detected Artifacts\n({len(metadata["artifacts_detected"])} found)', 
                             fontsize=12, fontweight='bold')
            else:
                ax6.imshow(original_image * 0.5)
                ax6.text(0.5, 0.5, 'No Artifacts\nDetected', ha='center', va='center',
                        transform=ax6.transAxes, fontsize=14, fontweight='bold', color='white')
                ax6.set_title('Artifact Analysis', fontsize=12, fontweight='bold')
            ax6.axis('off')
            
            # Quality metrics
            ax7 = fig.add_subplot(gs[1, 2])
            self._plot_quality_metrics(ax7, metadata.get('improvement_metrics', {}))
            
            # Processing steps
            ax8 = fig.add_subplot(gs[1, 3])
            self._plot_processing_steps(ax8, metadata.get('steps', []))
            
            # Row 3: Detailed analysis
            ax9 = fig.add_subplot(gs[2, :2])
            self._plot_intensity_analysis(ax9, original_image, original_mask, improved_mask)
            
            ax10 = fig.add_subplot(gs[2, 2:])
            self._plot_size_distribution(ax10, original_mask, improved_mask)
            
            # Overall title
            stain_type = metadata.get('image_properties', {}).get('stain_type', 'Unknown')
            fig.suptitle(f'Tissue Mask Improvement Analysis - {stain_type} Staining', 
                        fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            # Create a simple error plot instead of failing completely
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}\n\nImage shape: {original_image.shape if original_image is not None else "None"}\nMask shapes: {original_mask.shape if original_mask is not None else "None"}, {improved_mask.shape if improved_mask is not None else "None"}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax.set_title('Visualization Generation Failed', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
    
    def create_artifact_detection_plot(self, original_image: np.ndarray,
                                     artifacts: List[Dict[str, Any]],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a detailed plot showing detected artifacts.
        
        Args:
            original_image: Original histology image
            artifacts: List of detected artifacts
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image with artifact annotations
        ax = axes[0, 0]
        ax.imshow(original_image)
        self._annotate_artifacts(ax, artifacts)
        ax.set_title(f'Detected Artifacts ({len(artifacts)} total)', fontweight='bold')
        ax.axis('off')
        
        # Artifact type distribution
        ax = axes[0, 1]
        self._plot_artifact_distribution(ax, artifacts)
        
        # Artifact size analysis
        ax = axes[1, 0]
        self._plot_artifact_sizes(ax, artifacts)
        
        # Artifact location heatmap
        ax = axes[1, 1]
        self._plot_artifact_heatmap(ax, artifacts, original_image.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_batch_summary_plot(self, results: List[Dict[str, Any]], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a summary plot for batch processing results.
        
        Args:
            results: List of processing results for multiple images
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Extract metrics
        metrics = []
        for result in results:
            if 'improvement_metrics' in result:
                metrics.append(result['improvement_metrics'])
        
        if not metrics:
            fig.text(0.5, 0.5, 'No metrics available for batch summary', 
                    ha='center', va='center', fontsize=16)
            return fig
        
        # Area change distribution
        ax = axes[0, 0]
        area_changes = [m.get('area_change_ratio', 1.0) for m in metrics]
        ax.hist(area_changes, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', label='No change')
        ax.set_xlabel('Area Change Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Area Change Distribution')
        ax.legend()
        
        # Jaccard index distribution
        ax = axes[0, 1]
        jaccard_scores = [m.get('jaccard_index', 0.0) for m in metrics]
        ax.hist(jaccard_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('Jaccard Index')
        ax.set_ylabel('Frequency')
        ax.set_title('Overlap Quality Distribution')
        
        # Dice coefficient distribution
        ax = axes[0, 2]
        dice_scores = [m.get('dice_coefficient', 0.0) for m in metrics]
        ax.hist(dice_scores, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Dice Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Dice Coefficient Distribution')
        
        # Artifact reduction
        ax = axes[1, 0]
        artifact_reduction = [m.get('bright_artifact_reduction', 0.0) for m in metrics]
        ax.hist(artifact_reduction, bins=20, alpha=0.7, edgecolor='black', color='purple')
        ax.set_xlabel('Artifact Reduction Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Artifact Reduction Distribution')
        
        # Processing time if available
        ax = axes[1, 1]
        processing_times = []
        for result in results:
            if 'processing_time' in result:
                processing_times.append(result['processing_time'])
        
        if processing_times:
            ax.hist(processing_times, bins=20, alpha=0.7, edgecolor='black', color='red')
            ax.set_xlabel('Processing Time (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Processing Time Distribution')
        else:
            ax.text(0.5, 0.5, 'Processing time\nnot available', ha='center', va='center')
            ax.set_title('Processing Time Distribution')
        
        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Batch Summary Statistics
        ========================
        Total Images: {len(results)}
        
        Area Change:
        • Mean: {np.mean(area_changes):.3f}
        • Std: {np.std(area_changes):.3f}
        
        Jaccard Index:
        • Mean: {np.mean(jaccard_scores):.3f}
        • Std: {np.std(jaccard_scores):.3f}
        
        Dice Coefficient:
        • Mean: {np.mean(dice_scores):.3f}
        • Std: {np.std(dice_scores):.3f}
        
        Artifact Reduction:
        • Mean: {np.mean(artifact_reduction):.3f}
        • Std: {np.std(artifact_reduction):.3f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _create_mask_overlay(self, image: np.ndarray, mask: np.ndarray, 
                           alpha: float = 0.3) -> np.ndarray:
        """Create an overlay of the mask on the original image."""
        overlay = image.copy()
        
        # Create colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = mask  # Red channel for tissue
        
        # Blend with original image
        overlay = (1 - alpha) * overlay + alpha * mask_colored
        
        return np.clip(overlay, 0, 1)
    
    def _compute_mask_difference(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Compute the difference between two masks."""
        difference = mask2.astype(float) - mask1.astype(float)
        return difference
    
    def _visualize_artifacts(self, image: np.ndarray, artifacts: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize detected artifacts on the image."""
        viz_image = image.copy()
        
        for artifact in artifacts:
            bbox = artifact.get('bbox', None)
            if bbox:
                y1, x1, y2, x2 = bbox
                # Draw bounding box
                viz_image[y1:y1+2, x1:x2] = [1, 0, 0]  # Red top
                viz_image[y2-2:y2, x1:x2] = [1, 0, 0]  # Red bottom
                viz_image[y1:y2, x1:x1+2] = [1, 0, 0]  # Red left
                viz_image[y1:y2, x2-2:x2] = [1, 0, 0]  # Red right
        
        return viz_image
    
    def _annotate_artifacts(self, ax: plt.Axes, artifacts: List[Dict[str, Any]]):
        """Annotate artifacts on a plot."""
        colors = {'large_artifact': 'red', 'linear_artifact': 'yellow', 'spot_artifact': 'cyan'}
        
        for i, artifact in enumerate(artifacts):
            bbox = artifact.get('bbox', None)
            artifact_type = artifact.get('type', 'unknown')
            
            if bbox:
                y1, x1, y2, x2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                color = colors.get(artifact_type, 'white')
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1-5, f'{artifact_type}_{i}', color=color, 
                       fontsize=8, fontweight='bold')
    
    def _plot_quality_metrics(self, ax: plt.Axes, metrics: Dict[str, float]):
        """Plot quality metrics as a bar chart."""
        if not metrics:
            ax.text(0.5, 0.5, 'No metrics\navailable', ha='center', va='center')
            ax.set_title('Quality Metrics')
            return
        
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_names.append(key.replace('_', '\n'))
                metric_values.append(value)
        
        if metric_names:
            bars = ax.bar(metric_names, metric_values, alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title('Quality Metrics')
            ax.tick_params(axis='x', rotation=45)
            
            # Color code bars
            for i, bar in enumerate(bars):
                if metric_values[i] > 0.8:
                    bar.set_color('green')
                elif metric_values[i] > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
    
    def _plot_processing_steps(self, ax: plt.Axes, steps: List[str]):
        """Plot processing steps as a timeline."""
        ax.axis('off')
        
        if not steps:
            ax.text(0.5, 0.5, 'No steps\nrecorded', ha='center', va='center')
            ax.set_title('Processing Steps')
            return
        
        ax.set_title('Processing Pipeline', fontweight='bold')
        
        for i, step in enumerate(steps):
            y_pos = 0.9 - i * 0.15
            
            # Draw step box
            rect = patches.FancyBboxPatch((0.1, y_pos-0.05), 0.8, 0.1,
                                        boxstyle="round,pad=0.02", 
                                        facecolor='lightblue', 
                                        edgecolor='blue',
                                        alpha=0.7)
            ax.add_patch(rect)
            
            # Add step text
            ax.text(0.5, y_pos, f"{i+1}. {step.replace('_', ' ').title()}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add arrow if not last step
            if i < len(steps) - 1:
                ax.arrow(0.5, y_pos-0.08, 0, -0.04, head_width=0.02, 
                        head_length=0.02, fc='blue', ec='blue')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_intensity_analysis(self, ax: plt.Axes, image: np.ndarray, 
                               original_mask: np.ndarray, improved_mask: np.ndarray):
        """Plot intensity analysis comparison."""
        gray = rgb2gray(image)
        
        # Extract intensities for each region
        background_intensities = gray[~original_mask & ~improved_mask]
        original_tissue = gray[original_mask]
        improved_tissue = gray[improved_mask]
        added_regions = gray[improved_mask & ~original_mask]
        removed_regions = gray[original_mask & ~improved_mask]
        
        # Create histogram
        bins = np.linspace(0, 1, 50)
        
        if len(background_intensities) > 0:
            ax.hist(background_intensities, bins=bins, alpha=0.5, label='Background', color='gray')
        if len(original_tissue) > 0:
            ax.hist(original_tissue, bins=bins, alpha=0.5, label='Original Tissue', color='blue')
        if len(improved_tissue) > 0:
            ax.hist(improved_tissue, bins=bins, alpha=0.5, label='Improved Tissue', color='green')
        if len(added_regions) > 0:
            ax.hist(added_regions, bins=bins, alpha=0.7, label='Added Regions', color='cyan')
        if len(removed_regions) > 0:
            ax.hist(removed_regions, bins=bins, alpha=0.7, label='Removed Regions', color='red')
        
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Intensity Distribution Analysis')
        ax.legend()
    
    def _plot_size_distribution(self, ax: plt.Axes, original_mask: np.ndarray, 
                              improved_mask: np.ndarray):
        """Plot size distribution of connected components."""
        # Analyze original mask
        original_labeled = measure.label(original_mask)
        original_sizes = []
        if original_labeled.max() > 0:
            original_sizes = [region.area for region in measure.regionprops(original_labeled)]
        
        # Analyze improved mask
        improved_labeled = measure.label(improved_mask)
        improved_sizes = []
        if improved_labeled.max() > 0:
            improved_sizes = [region.area for region in measure.regionprops(improved_labeled)]
        
        # Create comparison plot
        if original_sizes:
            ax.hist(original_sizes, bins=20, alpha=0.7, label='Original', color='blue')
        if improved_sizes:
            ax.hist(improved_sizes, bins=20, alpha=0.7, label='Improved', color='green')
        
        ax.set_xlabel('Component Size (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Connected Component Size Distribution')
        ax.legend()
        ax.set_yscale('log')
    
    def _plot_artifact_distribution(self, ax: plt.Axes, artifacts: List[Dict[str, Any]]):
        """Plot distribution of artifact types."""
        if not artifacts:
            ax.text(0.5, 0.5, 'No artifacts\ndetected', ha='center', va='center')
            ax.set_title('Artifact Type Distribution')
            return
        
        artifact_types = [a.get('type', 'unknown') for a in artifacts]
        unique_types, counts = np.unique(artifact_types, return_counts=True)
        
        bars = ax.bar(unique_types, counts, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Artifact Type Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # Color code bars
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
    
    def _plot_artifact_sizes(self, ax: plt.Axes, artifacts: List[Dict[str, Any]]):
        """Plot artifact size distribution."""
        if not artifacts:
            ax.text(0.5, 0.5, 'No artifacts\ndetected', ha='center', va='center')
            ax.set_title('Artifact Size Distribution')
            return
        
        sizes = [a.get('area', 0) for a in artifacts]
        ax.hist(sizes, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Artifact Size (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Artifact Size Distribution')
        ax.set_yscale('log')
    
    def _plot_artifact_heatmap(self, ax: plt.Axes, artifacts: List[Dict[str, Any]], 
                             image_shape: Tuple[int, int]):
        """Plot heatmap of artifact locations."""
        h, w = image_shape
        heatmap = np.zeros((h, w))
        
        for artifact in artifacts:
            centroid = artifact.get('centroid', None)
            if centroid:
                y, x = int(centroid[0]), int(centroid[1])
                if 0 <= y < h and 0 <= x < w:
                    # Create gaussian blob around centroid
                    y_grid, x_grid = np.ogrid[:h, :w]
                    distance = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
                    heatmap += np.exp(-distance**2 / (2 * 50**2))
        
        if np.max(heatmap) > 0:
            im = ax.imshow(heatmap, cmap='hot', alpha=0.7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(np.zeros((h, w)), cmap='gray')
            ax.text(w//2, h//2, 'No artifacts\ndetected', ha='center', va='center', 
                   color='white', fontsize=14, fontweight='bold')
        
        ax.set_title('Artifact Location Heatmap')
        ax.axis('off') 