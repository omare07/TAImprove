#!/usr/bin/env python3
"""
Tissue Mask Improvement System

Main script for improving histology tissue masks using advanced image processing
techniques. This system takes original histology images and their corresponding
tissue masks and generates improved masks that better isolate true tissue regions.

Usage:
    python main.py --image_dir path/to/images --mask_dir path/to/masks --output_dir path/to/output
    
Author: AI Assistant
Date: 2024
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
from skimage.transform import resize

# Import our modules
from tissue_mask_improver import TissueMaskImprover
from visualization import TissueMaskVisualizer
from utils import (
    ImageLoader, FileManager, ConfigManager, 
    setup_logging, validate_paths
)


def process_single_image(image_path: Path, mask_path: Path, 
                        improver: TissueMaskImprover,
                        output_paths: Dict[str, Path],
                        visualizer: Optional[TissueMaskVisualizer] = None,
                        create_visualizations: bool = True) -> Dict[str, Any]:
    """
    Process a single image-mask pair.
    
    Args:
        image_path: Path to original image
        mask_path: Path to current tissue mask
        improver: TissueMaskImprover instance
        output_paths: Dictionary of output directory paths
        visualizer: Optional visualizer instance
        create_visualizations: Whether to create visualization plots
        
    Returns:
        Dictionary containing processing results and metadata
    """
    start_time = time.time()
    
    try:
        # Load image and mask
        improver.logger.info(f"Processing: {image_path.name}")
        original_image = ImageLoader.load_image(image_path)
        current_mask = ImageLoader.load_mask(mask_path)
        
        # Validate image and mask compatibility
        if original_image.shape[:2] != current_mask.shape[:2]:
            improver.logger.warning(f"Size mismatch for {image_path.name}: "
                                   f"Image {original_image.shape[:2]} vs Mask {current_mask.shape[:2]}")
            # Resize mask to match image
            current_mask = resize(current_mask.astype(float), original_image.shape[:2], 
                                anti_aliasing=False, preserve_range=True) > 0.5
            improver.logger.info(f"Resized mask to match image dimensions")
        
        # Log image properties for debugging
        improver.logger.debug(f"Image shape: {original_image.shape}, dtype: {original_image.dtype}, "
                             f"range: [{original_image.min():.3f}, {original_image.max():.3f}]")
        improver.logger.debug(f"Mask shape: {current_mask.shape}, dtype: {current_mask.dtype}, "
                             f"tissue pixels: {np.sum(current_mask)}")
        
        # Improve the mask
        improved_mask, metadata = improver.improve_mask(original_image, current_mask)
        
        # Save improved mask
        output_mask_path = output_paths['improved_masks'] / f"{image_path.stem}_improved{mask_path.suffix}"
        ImageLoader.save_mask(improved_mask, output_mask_path)
        
        # Create visualizations if requested
        viz_success = True
        viz_error = None
        
        if create_visualizations and visualizer:
            try:
                # Comparison plot
                comparison_path = output_paths['visualizations'] / f"{image_path.stem}_comparison.png"
                improver.logger.debug(f"Creating comparison plot: {comparison_path}")
                
                comparison_fig = visualizer.create_comparison_plot(
                    original_image, current_mask, improved_mask, metadata, comparison_path
                )
                
                # Explicitly save and close
                if comparison_fig:
                    comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
                    comparison_fig.clear()
                    plt.close(comparison_fig)
                    improver.logger.debug(f"✅ Comparison plot saved: {comparison_path}")
                else:
                    improver.logger.warning(f"❌ Comparison plot creation returned None")
                
                # Skip artifact plots (can cause hangs)
                improver.logger.debug("Skipping artifact plot creation to avoid hangs")
                        
            except Exception as viz_e:
                viz_success = False
                viz_error = str(viz_e)
                improver.logger.error(f"❌ Visualization failed for {image_path.name}: {viz_e}")
                improver.logger.debug(traceback.format_exc())
        
        processing_time = time.time() - start_time
        
        # Compile results
        result = {
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'output_mask_path': str(output_mask_path),
            'processing_time': processing_time,
            'success': True,
            'visualization_success': viz_success,
            'visualization_error': viz_error,
            'image_shape': original_image.shape,
            'mask_shape': current_mask.shape,
            'error': None,
            **metadata
        }
        
        improver.logger.info(f"Successfully processed {image_path.name} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Failed to process {image_path.name}: {str(e)}"
        improver.logger.error(error_msg)
        improver.logger.debug(traceback.format_exc())
        
        return {
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'processing_time': processing_time,
            'success': False,
            'visualization_success': False,
            'error': error_msg
        }


def process_batch(image_mask_pairs: List[tuple], 
                 config_manager: ConfigManager,
                 output_dir: Path,
                 num_workers: int = 1) -> List[Dict[str, Any]]:
    """
    Process a batch of image-mask pairs.
    
    Args:
        image_mask_pairs: List of (image_path, mask_path) tuples
        config_manager: Configuration manager instance
        output_dir: Output directory path
        
    Returns:
        List of processing results
    """
    # Setup logging
    logging_config = config_manager.get_logging_config()
    logger = setup_logging(logging_config)
    
    # Initialize components
    processing_config = config_manager.get_processing_config()
    improver = TissueMaskImprover(processing_config)
    
    visualization_config = config_manager.get_visualization_config()
    create_visualizations = visualization_config.get('create_comparison_plots', True)
    visualizer = TissueMaskVisualizer() if create_visualizations else None
    
    # Setup simplified output directory structure
    file_manager = FileManager(logger)
    output_paths = file_manager.create_output_structure(output_dir, subdirs=['improved_masks', 'visualizations'])
    
    logger.info(f"Starting batch processing of {len(image_mask_pairs)} image pairs")
    logger.info(f"Workers: {num_workers} ({'parallel' if num_workers > 1 else 'sequential'})")
    logger.info(f"Output directory: {output_dir}")
    
    # Process all pairs
    results = []
    successful_count = 0
    
    if num_workers == 1:
        # Sequential processing (safe mode)
        logger.info("🔒 Running sequential processing (safe mode)")
        
        for i, (image_path, mask_path) in enumerate(image_mask_pairs, 1):
            try:
                logger.info(f"🔄 Processing {i}/{len(image_mask_pairs)}: {image_path.name}")
                start_time = time.time()
                
                result = process_single_image(
                    image_path, mask_path, improver, output_paths,
                    visualizer, create_visualizations
                )
                
                processing_time = time.time() - start_time
                results.append(result)
                
                if result['success']:
                    successful_count += 1
                    logger.info(f"✅ Success: {image_path.name} ({processing_time:.2f}s)")
                else:
                    logger.error(f"❌ Failed: {image_path.name} - {result.get('error', 'Unknown error')}")
                
                # Force garbage collection and matplotlib cleanup after each image
                plt.close('all')  # Close any lingering matplotlib figures
                
                # Progress update
                if i % 5 == 0 or i == len(image_mask_pairs):
                    logger.info(f"📈 Progress: {i}/{len(image_mask_pairs)} ({successful_count} ✅, {i-successful_count} ❌)")
                    
            except Exception as process_e:
                logger.error(f"💥 Critical error processing {image_path.name}: {process_e}")
                logger.debug(traceback.format_exc())
                
                # Add failed result and continue
                results.append({
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                    'processing_time': 0,
                    'success': False,
                    'error': f"Critical processing error: {process_e}"
                })
    else:
        # Simple parallel processing (use with caution - may hang on some systems)
        logger.warning(f"⚠️  Using parallel processing with {num_workers} threads - this may hang on some systems!")
        logger.warning(f"⚠️  If the system hangs, press Ctrl+C and use --workers 1 for sequential processing")
        
        # Create a simple wrapper function
        def process_wrapper(args):
            image_path, mask_path = args
            try:
                # Each thread gets its own instances to avoid conflicts
                thread_improver = TissueMaskImprover(processing_config)
                thread_visualizer = TissueMaskVisualizer() if create_visualizations else None
                
                result = process_single_image(
                    image_path, mask_path, thread_improver, output_paths,
                    thread_visualizer, create_visualizations
                )
                
                return result
                
            except Exception as e:
                return {
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),  
                    'processing_time': 0,
                    'success': False,
                    'error': f"Thread processing failed: {e}"
                }
        
        # Simple ThreadPoolExecutor with shorter timeout
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            try:
                # Submit all tasks and collect results
                futures = [executor.submit(process_wrapper, (img_path, mask_path)) 
                          for img_path, mask_path in image_mask_pairs]
                
                # Collect results with shorter timeout
                for i, future in enumerate(futures, 1):
                    try:
                        result = future.result(timeout=180)  # 3 minute timeout per image
                        results.append(result)
                        
                        if result['success']:
                            successful_count += 1
                            
                        # Progress update
                        if i % 5 == 0 or i == len(image_mask_pairs):
                            logger.info(f"Progress: {i}/{len(image_mask_pairs)} ({successful_count} successful)")
                            
                    except Exception as e:
                        logger.error(f"Task {i} failed: {e}")
                        results.append({
                            'image_path': 'unknown',
                            'processing_time': 0,
                            'success': False,
                            'error': f"Task execution failed: {e}"
                        })
                        
            except Exception as e:
                logger.error(f"Parallel processing failed completely: {e}")
                logger.error("Falling back to sequential processing...")
                # Fall back to sequential processing
                num_workers = 1
    
    # If parallel processing failed, run sequential processing 
    if num_workers == 1 and not results:
        logger.info("🔄 Running sequential processing (safe mode)")
        for i, (image_path, mask_path) in enumerate(image_mask_pairs, 1):
            logger.info(f"Processing {i}/{len(image_mask_pairs)}: {image_path.name}")
            
            result = process_single_image(
                image_path, mask_path, improver, output_paths,
                visualizer, create_visualizations
            )
            
            results.append(result)
            
            if result['success']:
                successful_count += 1
            
            # Progress update
            if i % 10 == 0 or i == len(image_mask_pairs):
                logger.info(f"Progress: {i}/{len(image_mask_pairs)} ({successful_count} successful)")
    
    # Skip batch summary visualization (can cause hangs)
    logger.info("Skipping batch summary visualization to avoid hangs")
    
    # Skip report saving (can cause hangs on network drives)
    logger.info("Skipping report generation to avoid network I/O hangs")
    
    logger.info(f"Batch processing completed: {successful_count}/{len(image_mask_pairs)} successful")
    return results


def main():
    """Main entry point for the tissue mask improvement system."""
    
    parser = argparse.ArgumentParser(
        description="Improve histology tissue masks using advanced image processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (RECOMMENDED - sequential processing)
    python main.py --image_dir "path/to/images" --mask_dir "path/to/masks" --output_dir "path/to/output"
    
    # With custom configuration (sequential)
    python main.py --image_dir "images" --mask_dir "masks" --output_dir "output" --config "config.json"
    
    # Sequential without visualizations (faster)
    python main.py --image_dir "images" --mask_dir "masks" --output_dir "output" --no_visualizations
    
    # Parallel processing (USE WITH CAUTION - may hang!)
    python main.py --image_dir "images" --mask_dir "masks" --output_dir "output" --workers 4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--image_dir', 
        type=str, 
        required=True,
        help='Directory containing original histology images'
    )
    
    parser.add_argument(
        '--mask_dir', 
        type=str, 
        required=True,
        help='Directory containing current tissue masks to improve'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save improved masks and results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--image_ext', 
        nargs='+', 
        default=['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
        help='Image file extensions to process (default: .tif .tiff .png .jpg .jpeg)'
    )
    
    parser.add_argument(
        '--mask_ext', 
        nargs='+', 
        default=['.tif', '.tiff', '.png'],
        help='Mask file extensions to process (default: .tif .tiff .png)'
    )
    
    parser.add_argument(
        '--no_visualizations', 
        action='store_true',
        help='Disable creation of visualization plots'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log_file', 
        type=str,
        help='Path to log file (optional)'
    )
    
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='Number of parallel workers (default: 1 for stability). WARNING: Parallel processing may hang on some systems!'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate input paths
        image_dir, mask_dir = validate_paths(args.image_dir, args.mask_dir)
        output_dir = Path(args.output_dir)
        
        # Initialize configuration
        config_manager = ConfigManager(args.config)
        
        # Override config with command line arguments
        if args.no_visualizations:
            viz_config = config_manager.get_visualization_config()
            viz_config.update({
                'create_comparison_plots': False,
                'create_artifact_plots': False,
                'create_batch_summary': False
            })
        
        if args.verbose:
            logging_config = config_manager.get_logging_config()
            logging_config['verbose'] = True
            logging_config['log_level'] = 'DEBUG'
        
        if args.log_file:
            logging_config = config_manager.get_logging_config()
            logging_config['log_file'] = args.log_file
        
        # Update file extensions in config
        file_config = config_manager.get_file_config()
        file_config['image_extensions'] = args.image_ext
        file_config['mask_extensions'] = args.mask_ext
        
        # Setup logging
        logging_config = config_manager.get_logging_config()
        logger = setup_logging(logging_config)
        
        logger.info("=" * 60)
        logger.info("Tissue Mask Improvement System")
        logger.info("=" * 60)
        
        if args.workers == 1:
            logger.info("🔒 Using SEQUENTIAL processing (safe mode - recommended)")
        else:
            logger.warning(f"⚠️  Using PARALLEL processing with {args.workers} workers")
            logger.warning("⚠️  This may cause hanging issues! Use --workers 1 if problems occur")
            
        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Mask directory: {mask_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Find image-mask pairs
        file_manager = FileManager(logger)
        pairs = file_manager.find_image_mask_pairs(
            image_dir, mask_dir, args.image_ext, args.mask_ext
        )
        
        if not pairs:
            logger.error("No matching image-mask pairs found!")
            sys.exit(1)
        
        logger.info(f"Found {len(pairs)} image-mask pairs to process")
        
        # Process the batch
        start_time = time.time()
        results = process_batch(pairs, config_manager, output_dir, args.workers)
        total_time = time.time() - start_time
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average time per image: {total_time/len(results):.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        
        if failed > 0:
            logger.warning(f"{failed} images failed to process. Check the processing report for details.")
            sys.exit(1)
        else:
            logger.info("All images processed successfully!")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 