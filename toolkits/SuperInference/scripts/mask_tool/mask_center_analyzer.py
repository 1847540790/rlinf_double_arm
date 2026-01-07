#!/usr/bin/env python3
"""
Mask Center Analyzer

This script analyzes mask images to calculate the center position of white regions
and the distance from the mask center to the image center point.

Features:
- Load mask images and detect white regions
- Calculate centroid of white regions (mask center)
- Calculate image center point with optional Y offset
- Compute distance between mask center and image center
- Create centered mask by moving white regions to the offset center
- Visualize results with annotations
- Save centered mask as PNG image

Usage:
    python scripts/mask_tool/mask_center_analyzer.py --mask_path calibration/masks/umi/mapped/left_mask_400*300.png
    python mask_center_analyzer.py --mask_path masks/left_mask.png --output_dir analysis_results
    python mask_center_analyzer.py --mask_path masks/left_mask.png --center_offset_y 10.0

Author: Assistant
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
import json
import time
import math

# Get the project root directory (parent of scripts directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from utils.logger_config import logger


class MaskCenterAnalyzer:
    """Analyzer for calculating mask center positions and distances."""

    def __init__(self, mask_path: Path, output_dir: Optional[Path] = None,
                 output_filename: Optional[str] = None, center_offset_y: float = 0.0):
        """
        Initialize the mask center analyzer.

        Args:
            mask_path: Path to the mask image file
            output_dir: Directory to save analysis results (default: mask_analysis)
            output_filename: Custom filename for JSON output (without extension)
            center_offset_y: Offset to shift the image center downward (positive values)
        """
        self.mask_path = Path(mask_path)
        self.output_dir = output_dir or Path("mask_analysis")
        self.output_filename = output_filename
        self.center_offset_y = center_offset_y

        # Mask data
        self.mask_image = None
        self.mask_binary = None

        # Analysis results
        self.mask_center = None
        self.image_center = None
        self.distance = None
        self.analysis_results = {}

        # Centered mask data
        self.centered_mask = None

        logger.info(f"Initializing mask center analyzer")
        logger.info(f"Mask path: {self.mask_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_mask(self) -> bool:
        """
        Load mask image and convert to binary format.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.mask_path.exists():
                logger.error(f"Mask file not found: {self.mask_path}")
                return False

            logger.info(f"Loading mask from {self.mask_path}")

            # Load mask image
            self.mask_image = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if self.mask_image is None:
                logger.error(f"Failed to load mask from {self.mask_path}")
                return False

            # Convert to binary mask (white regions = 1, black regions = 0)
            # Assuming white regions are the mask areas we want to analyze
            self.mask_binary = (self.mask_image > 127).astype(np.uint8)

            logger.info(f"Mask image shape: {self.mask_image.shape}")
            logger.info(f"Mask binary shape: {self.mask_binary.shape}")
            logger.info(f"White pixel count: {np.sum(self.mask_binary)}")
            logger.info(f"White pixel percentage: {np.sum(self.mask_binary) / self.mask_binary.size * 100:.2f}%")

            return True

        except Exception as e:
            logger.error(f"Error loading mask: {e}")
            return False

    def calculate_mask_center(self) -> Tuple[float, float]:
        """
        Calculate the centroid of white regions in the mask.

        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        try:
            if self.mask_binary is None:
                logger.error("No mask loaded")
                return 0.0, 0.0

            # Calculate moments to find centroid
            moments = cv2.moments(self.mask_binary)

            if moments['m00'] == 0:
                logger.error("No white regions found in mask")
                return 0.0, 0.0

            # Calculate centroid coordinates
            center_x = moments['m10'] / moments['m00']
            center_y = moments['m01'] / moments['m00']

            self.mask_center = (center_x, center_y)

            logger.info(f"Mask center (centroid): ({center_x:.2f}, {center_y:.2f})")

            return center_x, center_y

        except Exception as e:
            logger.error(f"Error calculating mask center: {e}")
            return 0.0, 0.0

    def calculate_image_center(self) -> Tuple[float, float]:
        """
        Calculate the center point of the image with optional Y offset.

        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        try:
            if self.mask_image is None:
                logger.error("No mask image loaded")
                return 0.0, 0.0

            height, width = self.mask_image.shape
            center_x = width / 2.0
            center_y = height / 2.0 + self.center_offset_y

            self.image_center = (center_x, center_y)

            logger.info(f"Image center: ({center_x:.2f}, {center_y:.2f})")
            logger.info(f"Image dimensions: {width}x{height}")
            if self.center_offset_y != 0.0:
                logger.info(f"Applied Y offset: {self.center_offset_y:.2f}")

            return center_x, center_y

        except Exception as e:
            logger.error(f"Error calculating image center: {e}")
            return 0.0, 0.0

    def calculate_distance(self) -> float:
        """
        Calculate the distance between mask center and image center.

        Returns:
            Distance in pixels
        """
        try:
            if self.mask_center is None or self.image_center is None:
                logger.error("Mask center or image center not calculated")
                return 0.0

            # Calculate Euclidean distance
            dx = self.mask_center[0] - self.image_center[0]
            dy = self.mask_center[1] - self.image_center[1]
            distance = math.sqrt(dx * dx + dy * dy)

            self.distance = distance

            logger.info(f"Distance from mask center to image center: {distance:.2f} pixels")
            logger.info(f"Offset: X={dx:.2f}, Y={dy:.2f}")

            return distance

        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0

    def create_centered_mask(self) -> bool:
        """
        Create a new mask with the white region moved to the offset image center.

        Returns:
            True if successful, False otherwise
        """
        try:
            if (self.mask_center is None or self.image_center is None or
                    self.mask_binary is None):
                logger.error("Required data not available for creating centered mask")
                return False

            # Calculate offset to move mask center to the offset image center
            offset_x = self.image_center[0] - self.mask_center[0]
            offset_y = self.image_center[1] - self.mask_center[1]

            logger.info(f"Creating centered mask with offset: ({offset_x:.2f}, {offset_y:.2f})")
            logger.info(f"Moving mask center from ({self.mask_center[0]:.2f}, {self.mask_center[1]:.2f}) to ({self.image_center[0]:.2f}, {self.image_center[1]:.2f})")

            # Create transformation matrix for translation
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

            # Apply transformation to move mask to center
            height, width = self.mask_binary.shape
            self.centered_mask = cv2.warpAffine(
                self.mask_binary, M, (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Convert back to 0-255 range for saving
            self.centered_mask = (self.centered_mask * 255).astype(np.uint8)

            logger.info("Centered mask created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating centered mask: {e}")
            return False

    def analyze_mask_properties(self) -> Dict[str, Any]:
        """
        Analyze additional properties of the mask.

        Returns:
            Dictionary with mask properties
        """
        try:
            if self.mask_binary is None:
                return {}

            # Calculate mask area
            mask_area = np.sum(self.mask_binary)
            total_area = self.mask_binary.size
            coverage_percentage = (mask_area / total_area) * 100

            # Find bounding box of white regions
            coords = np.where(self.mask_binary > 0)
            if len(coords[0]) > 0:
                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()
                bbox_width = max_x - min_x + 1
                bbox_height = max_y - min_y + 1
                bbox_area = bbox_width * bbox_height
            else:
                min_x = min_y = max_x = max_y = 0
                bbox_width = bbox_height = bbox_area = 0

            # Calculate circularity (how close to a circle)
            if mask_area > 0:
                # Find contours
                contours, _ = cv2.findContours(self.mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_area = cv2.contourArea(largest_contour)
                    contour_perimeter = cv2.arcLength(largest_contour, True)

                    # Circularity = 4π * area / perimeter² (1.0 for perfect circle)
                    if contour_perimeter > 0:
                        circularity = 4 * math.pi * contour_area / (contour_perimeter * contour_perimeter)
                    else:
                        circularity = 0.0
                else:
                    circularity = 0.0
            else:
                circularity = 0.0

            properties = {
                'mask_area': int(mask_area),
                'total_area': int(total_area),
                'coverage_percentage': float(coverage_percentage),
                'bounding_box': {
                    'min_x': int(min_x),
                    'min_y': int(min_y),
                    'max_x': int(max_x),
                    'max_y': int(max_y),
                    'width': int(bbox_width),
                    'height': int(bbox_height),
                    'area': int(bbox_area)
                },
                'circularity': float(circularity),
                'image_dimensions': {
                    'width': int(self.mask_image.shape[1]),
                    'height': int(self.mask_image.shape[0])
                }
            }

            logger.info(f"Mask properties:")
            logger.info(f"  Coverage: {coverage_percentage:.2f}%")
            logger.info(f"  Bounding box: {bbox_width}x{bbox_height}")
            logger.info(f"  Circularity: {circularity:.3f} (1.0 = perfect circle)")

            return properties

        except Exception as e:
            logger.error(f"Error analyzing mask properties: {e}")
            return {}

    def create_visualization(self) -> np.ndarray:
        """
        Create visualization image with annotations.

        Returns:
            Visualization image
        """
        try:
            if self.mask_image is None:
                logger.error("No mask image loaded")
                return None

            # Create a larger canvas to show both original and centered masks
            height, width = self.mask_image.shape
            canvas_width = width * 2 + 50  # Two images side by side with gap
            canvas_height = height + 100  # Extra space for text

            # Create canvas
            vis_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Left panel: Original mask
            original_colored = cv2.cvtColor(self.mask_image, cv2.COLOR_GRAY2BGR)
            vis_image[50:50 + height, 0:width] = original_colored

            # Right panel: Centered mask (if available)
            if self.centered_mask is not None:
                centered_colored = cv2.cvtColor(self.centered_mask, cv2.COLOR_GRAY2BGR)
                vis_image[50:50 + height, width + 50:2 * width + 50] = centered_colored
            else:
                # Show placeholder if centered mask not available
                placeholder = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Centered Mask", (width // 4, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                vis_image[50:50 + height, width + 50:2 * width + 50] = placeholder

            # Draw mask center on original image (left panel)
            if self.mask_center is not None:
                center_x, center_y = int(self.mask_center[0]), int(self.mask_center[1])
                cv2.circle(vis_image, (center_x, center_y + 50), 8, (0, 255, 0), -1)  # Green circle
                cv2.circle(vis_image, (center_x, center_y + 50), 12, (0, 255, 0), 2)  # Green outline
                cv2.putText(vis_image, "Mask Center", (center_x + 15, center_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw image center on both panels
            if self.image_center is not None:
                img_center_x, img_center_y = int(self.image_center[0]), int(self.image_center[1])

                # Left panel (original)
                cv2.circle(vis_image, (img_center_x, img_center_y + 50), 8, (0, 0, 255), -1)  # Red circle
                cv2.circle(vis_image, (img_center_x, img_center_y + 50), 12, (0, 0, 255), 2)  # Red outline
                cv2.putText(vis_image, "Image Center", (img_center_x + 15, img_center_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Right panel (centered mask)
                if self.centered_mask is not None:
                    cv2.circle(vis_image, (img_center_x + width + 50, img_center_y + 50), 8, (0, 0, 255),
                               -1)  # Red circle
                    cv2.circle(vis_image, (img_center_x + width + 50, img_center_y + 50), 12, (0, 0, 255),
                               2)  # Red outline
                    cv2.putText(vis_image, "Image Center", (img_center_x + width + 65, img_center_y + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw line connecting the two centers on left panel
            if (self.mask_center is not None and self.image_center is not None and
                    self.distance is not None):
                cv2.line(vis_image,
                         (int(self.mask_center[0]), int(self.mask_center[1]) + 50),
                         (int(self.image_center[0]), int(self.image_center[1]) + 50),
                         (255, 255, 0), 2)  # Yellow line

                # Add distance text
                mid_x = int((self.mask_center[0] + self.image_center[0]) / 2)
                mid_y = int((self.mask_center[1] + self.image_center[1]) / 2) + 50
                cv2.putText(vis_image, f"Distance: {self.distance:.1f}px",
                            (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Add panel labels
            cv2.putText(vis_image, "Original Mask", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_image, "Centered Mask", (width + 60, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add information panel
            self._add_info_panel(vis_image)

            return vis_image

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None

    def _add_info_panel(self, vis_image: np.ndarray):
        """Add information panel to visualization image."""
        try:
            height, width = vis_image.shape[:2]

            # Create semi-transparent overlay
            overlay = vis_image.copy()

            # Add background rectangle for text (positioned at bottom)
            info_y = height - 80
            cv2.rectangle(overlay, (10, info_y), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)

            # Add text information
            y_offset = info_y + 20
            line_height = 15

            if self.mask_center is not None:
                cv2.putText(vis_image, f"Mask Center: ({self.mask_center[0]:.1f}, {self.mask_center[1]:.1f})",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += line_height

            if self.image_center is not None:
                cv2.putText(vis_image, f"Image Center: ({self.image_center[0]:.1f}, {self.image_center[1]:.1f})",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += line_height

            if self.distance is not None:
                cv2.putText(vis_image, f"Distance: {self.distance:.2f} pixels",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += line_height

            # Add image dimensions
            cv2.putText(vis_image, f"Image Size: {width}x{height}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            logger.error(f"Error adding info panel: {e}")

    def save_results(self) -> bool:
        """
        Save analysis results to JSON file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare results data
            results_data = {
                'analysis_info': {
                    'mask_path': str(self.mask_path),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_dimensions': self.mask_image.shape if self.mask_image is not None else None
                },
                'mask_center': {
                    'x': float(self.mask_center[0]) if self.mask_center else None,
                    'y': float(self.mask_center[1]) if self.mask_center else None
                },
                'image_center': {
                    'x': float(self.image_center[0]) if self.image_center else None,
                    'y': float(self.image_center[1]) if self.image_center else None
                },
                'distance': float(self.distance) if self.distance else None,
                'offset': {
                    'x': float(self.mask_center[0] - self.image_center[0]) if (
                                self.mask_center and self.image_center) else None,
                    'y': float(self.mask_center[1] - self.image_center[1]) if (
                                self.mask_center and self.image_center) else None
                },
                'mask_properties': self.analysis_results.get('mask_properties', {})
            }

            # Save to JSON file
            if self.output_filename:
                # Use custom filename
                filename = f"{self.output_filename}.json"
            else:
                # Generate filename based on mask name and timestamp
                mask_name = self.mask_path.stem  # Get filename without extension
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{mask_name}_center_analysis_{timestamp}.json"

            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Analysis results saved to {filepath}")

            # Also save a summary text file
            if self.output_filename:
                summary_filename = f"{self.output_filename}_summary.txt"
            else:
                mask_name = self.mask_path.stem
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                summary_filename = f"{mask_name}_center_summary_{timestamp}.txt"

            summary_filepath = self.output_dir / summary_filename

            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("Mask Center Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Mask File: {self.mask_path}\n")
                f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if self.mask_center:
                    f.write(f"Mask Center: ({self.mask_center[0]:.2f}, {self.mask_center[1]:.2f})\n")
                if self.image_center:
                    f.write(f"Image Center: ({self.image_center[0]:.2f}, {self.image_center[1]:.2f})\n")
                if self.distance is not None:
                    f.write(f"Distance: {self.distance:.2f} pixels\n")

                if self.mask_center and self.image_center:
                    offset_x = self.mask_center[0] - self.image_center[0]
                    offset_y = self.mask_center[1] - self.image_center[1]
                    f.write(f"Offset: X={offset_x:.2f}, Y={offset_y:.2f}\n")

            logger.info(f"Summary saved to {summary_filepath}")

            # Save centered mask as PNG
            if self.centered_mask is not None:
                if self.output_filename:
                    centered_filename = f"{self.output_filename}_centered.png"
                else:
                    mask_name = self.mask_path.stem
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    centered_filename = f"{mask_name}_centered_{timestamp}.png"

                centered_filepath = self.output_dir / centered_filename
                cv2.imwrite(str(centered_filepath), self.centered_mask)
                logger.info(f"Centered mask saved to {centered_filepath}")

            return True

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False

    def visualize_results(self, show_visualization: bool = True, save_visualization: bool = True) -> bool:
        """
        Visualize analysis results.

        Args:
            show_visualization: Whether to show interactive visualization
            save_visualization: Whether to save visualization images

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create visualization
            vis_image = self.create_visualization()
            if vis_image is None:
                logger.error("Failed to create visualization")
                return False

            if show_visualization:
                # Show visualization
                cv2.namedWindow("Mask Center Analysis", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mask Center Analysis", 800, 600)
                cv2.imshow("Mask Center Analysis", vis_image)

                logger.info("Press any key to close visualization window")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_visualization:
                # Save visualization
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if self.output_filename:
                    filename = f"{self.output_filename}_visualization.png"
                else:
                    mask_name = self.mask_path.stem
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{mask_name}_center_visualization_{timestamp}.png"

                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), vis_image)
                logger.info(f"Visualization saved to {filepath}")

            return True

        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return False

    def run(self) -> bool:
        """
        Run the complete mask center analysis pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting mask center analysis")

        # Load mask
        if not self.load_mask():
            logger.error("Failed to load mask")
            return False

        # Calculate mask center
        if not self.calculate_mask_center():
            logger.error("Failed to calculate mask center")
            return False

        # Calculate image center
        if not self.calculate_image_center():
            logger.error("Failed to calculate image center")
            return False

        # Calculate distance
        if not self.calculate_distance():
            logger.error("Failed to calculate distance")
            return False

        # Create centered mask
        if not self.create_centered_mask():
            logger.error("Failed to create centered mask")
            return False

        # Analyze mask properties
        mask_properties = self.analyze_mask_properties()
        self.analysis_results['mask_properties'] = mask_properties

        # Visualize results
        if not self.visualize_results(show_visualization=True, save_visualization=True):
            logger.error("Failed to visualize results")
            return False

        # Save results
        if not self.save_results():
            logger.error("Failed to save results")
            return False

        # Print summary
        self._print_summary()

        logger.info("Mask center analysis completed successfully")
        return True

    def _print_summary(self):
        """Print analysis summary to console."""
        print("\n" + "=" * 60)
        print("MASK CENTER ANALYSIS SUMMARY")
        print("=" * 60)

        if self.mask_center:
            print(f"Mask Center (White Region Centroid): ({self.mask_center[0]:.2f}, {self.mask_center[1]:.2f})")

        if self.image_center:
            print(f"Image Center: ({self.image_center[0]:.2f}, {self.image_center[1]:.2f})")

        if self.distance is not None:
            print(f"Distance: {self.distance:.2f} pixels")

        if self.mask_center and self.image_center:
            offset_x = self.mask_center[0] - self.image_center[0]
            offset_y = self.mask_center[1] - self.image_center[1]
            print(f"Offset: X={offset_x:.2f}, Y={offset_y:.2f}")

        if 'mask_properties' in self.analysis_results:
            props = self.analysis_results['mask_properties']
            if 'coverage_percentage' in props:
                print(f"White Region Coverage: {props['coverage_percentage']:.2f}%")
            if 'circularity' in props:
                print(f"Circularity: {props['circularity']:.3f} (1.0 = perfect circle)")

        print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze mask center positions and calculate distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_center_analyzer.py --mask_path calibration/masks/umi/mapped/left_mask_400*300.png
  python mask_center_analyzer.py --mask_path masks/left_mask.png --output_dir analysis_results
  python mask_center_analyzer.py --mask_path masks/left_mask.png --output_filename left_mask_analysis
  python mask_center_analyzer.py --mask_path masks/left_mask.png --output_dir results --output_filename my_analysis
  python mask_center_analyzer.py --mask_path masks/left_mask.png --center_offset_y 10.0
        """
    )

    parser.add_argument(
        "--mask_path",
        type=Path,
        required=True,
        help="Path to the mask image file"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save analysis results (default: mask_analysis)"
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Custom filename for output files (without extension, e.g., 'left_mask_analysis')"
    )

    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Skip interactive visualization"
    )

    parser.add_argument(
        "--center_offset_y",
        type=float,
        default=0.0,
        help="Offset to shift the image center downward (positive values) (default: 0.0)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.mask_path.exists():
        logger.error(f"Mask file not found: {args.mask_path}")
        sys.exit(1)

    # Create and run analyzer
    analyzer = MaskCenterAnalyzer(
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        center_offset_y=args.center_offset_y
    )

    success = analyzer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()