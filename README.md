# Tissue Mask Improvement System

An advanced Python system for improving histology tissue masks using sophisticated image processing techniques. This system analyzes original histology images and their corresponding binary tissue masks to generate refined masks that accurately isolate true tissue regions while removing artifacts such as air bubbles, glare streaks, and background noise.

## Features

- **Intelligent Artifact Detection**: Automatically detects and removes common histology artifacts including:
  - Air bubbles (circular, low-texture regions)
  - Glare streaks (bright, elongated regions)
  - Transparent/poorly stained regions
  - Edge artifacts and background noise

- **Multi-Stain Support**: Handles both H&E and IHC stained images with stain-specific processing algorithms

- **Advanced Segmentation**: Uses multiple image features for robust tissue identification:
  - Intensity-based analysis
  - Texture analysis using entropy and local binary patterns
  - Color variance and saturation analysis
  - Edge detection and morphological operations

- **Comprehensive Visualization**: Generates detailed analysis plots including:
  - Before/after mask comparisons
  - Artifact detection visualizations
  - Quality metrics and processing pipeline diagrams
  - Batch processing summaries

- **Batch Processing**: Efficiently processes large datasets with progress tracking and error handling

- **Configurable Pipeline**: Fully configurable processing parameters via JSON configuration files

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by running:
   ```bash
   python main.py --help
   ```

## Quick Start

### Basic Usage

Process a directory of histology images and their corresponding tissue masks:

```bash
python main.py \
    --image_dir "path/to/original/images" \
    --mask_dir "path/to/current/masks" \
    --output_dir "path/to/output"
```

### Using Your Provided Paths

Based on your directory structure:

```bash
python main.py \
    --image_dir "\\\\10.99.134.183\\kiemen-lab-data\\exchange_data\\Omar Elfernani\\KPC6-1\\chop\\1x" \
    --mask_dir "\\\\10.99.134.183\\kiemen-lab-data\\exchange_data\\Omar Elfernani\\KPC6-1\\chop\\1x\\TA\\improve" \
    --output_dir "C:\\path\\to\\output"
```

## Command Line Options

### Required Arguments
- `--image_dir`: Directory containing original histology images
- `--mask_dir`: Directory containing current tissue masks to improve  
- `--output_dir`: Directory to save improved masks and results

### Optional Arguments
- `--config`: Path to JSON configuration file
- `--image_ext`: Image file extensions to process (default: .tif .tiff .png .jpg .jpeg)
- `--mask_ext`: Mask file extensions to process (default: .tif .tiff .png)
- `--no_visualizations`: Disable visualization creation for faster processing
- `--verbose`: Enable detailed logging
- `--log_file`: Save logs to specified file

### Examples

**Process only TIFF files with custom configuration:**
```bash
python main.py \
    --image_dir "images" \
    --mask_dir "masks" \
    --output_dir "output" \
    --config "custom_config.json" \
    --image_ext ".tif" ".tiff"
```

**Fast processing without visualizations:**
```bash
python main.py \
    --image_dir "images" \
    --mask_dir "masks" \
    --output_dir "output" \
    --no_visualizations \
    --verbose
```

## File Structure

The system expects the following file organization:

```
image_dir/
├── image1.tif
├── image2.tif
└── ...

mask_dir/
├── image1.tif        # Corresponding mask for image1.tif
├── image2.tif        # Corresponding mask for image2.tif
└── ...
```

The system automatically pairs images with their corresponding masks based on filename (excluding extension).

## Output Structure

The system creates the following output structure:

```
output_dir/
├── improved_masks/           # Improved tissue masks
│   ├── image1_improved.tif
│   └── image2_improved.tif
├── visualizations/           # Analysis plots
│   ├── image1_comparison.png
│   ├── image2_comparison.png
│   ├── image1_artifacts.png  # If artifacts detected
│   └── batch_summary.png
├── reports/                  # Processing reports
│   └── processing_report.json
└── intermediate/             # Intermediate results (if enabled)
```

## Configuration

### Default Configuration

The system comes with sensible defaults, but you can customize processing parameters using a JSON configuration file:

```json
{
  "processing": {
    "min_tissue_intensity": 10,
    "max_background_intensity": 250,
    "close_kernel_size": 3,
    "open_kernel_size": 2,
    "min_object_area": 1000,
    "bubble_circularity_threshold": 0.8,
    "glare_intensity_threshold": 240,
    "he_green_threshold": 210
  },
  "visualization": {
    "create_comparison_plots": true,
    "create_artifact_plots": true,
    "create_batch_summary": true,
    "plot_dpi": 300
  },
  "logging": {
    "verbose": true,
    "log_level": "INFO"
  }
}
```

### Creating Custom Configuration

1. **Generate default config:**
   ```python
   from utils import ConfigManager
   config = ConfigManager()
   config.save_config("my_config.json")
   ```

2. **Edit the JSON file** with your preferred parameters

3. **Use the config:**
   ```bash
   python main.py --config "my_config.json" [other arguments]
   ```

## Algorithm Overview

The tissue mask improvement system follows this processing pipeline:

1. **Image Analysis**: Analyzes input image properties (staining type, background estimation, saturation)

2. **Artifact Detection**: Identifies various artifacts using shape, texture, and intensity analysis

3. **Enhanced Segmentation**: Performs multi-feature tissue segmentation using:
   - Intensity deviation from background
   - Texture analysis (entropy, edge density)
   - Color variance analysis
   - Morphological operations

4. **Mask Combination**: Intelligently combines original mask, enhanced segmentation, and artifact removal

5. **Morphological Refinement**: Applies closing, opening, and size filtering operations

6. **Quality Control**: Validates results and applies fallback methods if needed

## Quality Metrics

The system provides several metrics to quantify improvement:

- **Jaccard Index**: Overlap between original and improved masks
- **Dice Coefficient**: Similarity measure between masks
- **Area Change Ratio**: Proportional change in tissue area
- **Artifact Reduction**: Estimated reduction in bright artifacts

## Comparison with Original MATLAB Code

This Python implementation improves upon the original MATLAB code by:

- **Enhanced Artifact Detection**: More sophisticated detection of air bubbles, glare, and transparent regions
- **Multi-Feature Segmentation**: Uses texture, color, and edge information in addition to intensity
- **Better Morphological Operations**: Adaptive kernel sizes and intelligent hole filling
- **Comprehensive Visualization**: Detailed analysis plots for quality assessment
- **Robust Error Handling**: Graceful handling of problematic images
- **Batch Processing**: Efficient processing of large datasets
- **Configurable Parameters**: Easy customization without code modification

## Troubleshooting

### Common Issues

**1. "No matching image-mask pairs found"**
- Check that image and mask files have the same base names
- Verify file extensions match the expected formats
- Use `--image_ext` and `--mask_ext` to specify custom extensions

**2. Memory errors with large images**
- Process images in smaller batches
- Reduce `plot_dpi` in configuration
- Disable visualizations with `--no_visualizations`

**3. Poor segmentation results**
- Adjust `min_tissue_intensity` and other thresholds in configuration
- Check if images are H&E or IHC stained (algorithm adapts automatically)
- Review visualization plots to understand processing decisions

**4. Network path issues (Windows)**
- Use forward slashes or escape backslashes in paths
- Consider mapping network drives to local drive letters

### Performance Tips

- Use `--no_visualizations` for faster processing
- Process TIFF files directly (avoid format conversion)
- Use SSD storage for better I/O performance
- For very large datasets, consider processing in batches

## Technical Requirements

- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum, 8GB+ recommended for large images
- **Storage**: Sufficient space for output (approximately 2x input size)
- **OS**: Windows, macOS, or Linux

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the generated log files and processing reports
3. Enable verbose logging with `--verbose` for detailed information
4. Check that all dependencies are correctly installed

## License

This software is provided as-is for research and educational purposes. 