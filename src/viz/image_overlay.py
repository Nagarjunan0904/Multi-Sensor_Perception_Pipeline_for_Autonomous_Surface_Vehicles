"""
Image overlay visualization utilities.

Production-quality 2D visualization functions for object detection results,
point cloud projections, and annotation overlays.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# =============================================================================
# Color Palettes (RGB format)
# =============================================================================

# KITTI object classes color palette
KITTI_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Car": (0, 255, 127),           # Spring green
    "Pedestrian": (255, 82, 82),    # Coral red
    "Cyclist": (64, 156, 255),      # Dodger blue
    "Van": (0, 206, 209),           # Dark turquoise
    "Truck": (255, 193, 37),        # Golden
    "Person_sitting": (255, 105, 180),  # Hot pink
    "Tram": (148, 0, 211),          # Dark violet
    "Misc": (169, 169, 169),        # Dark gray
    "DontCare": (105, 105, 105),    # Dim gray
}

# General purpose color palette (for unlabeled detections)
DETECTION_PALETTE = [
    (0, 255, 127),    # Spring green
    (255, 82, 82),    # Coral red
    (64, 156, 255),   # Dodger blue
    (255, 193, 37),   # Golden
    (0, 206, 209),    # Dark turquoise
    (255, 105, 180),  # Hot pink
    (148, 0, 211),    # Dark violet
    (50, 205, 50),    # Lime green
    (255, 140, 0),    # Dark orange
    (30, 144, 255),   # Dodger blue
]

# Depth colormap: blue (near) -> cyan -> green -> yellow -> red (far)
DEPTH_COLORMAP = cv2.COLORMAP_TURBO


def get_color_for_class(
    class_name: str,
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Tuple[int, int, int]:
    """
    Get color for a class name.

    Args:
        class_name: Object class name.
        class_colors: Optional custom color mapping.

    Returns:
        RGB color tuple.
    """
    if class_colors and class_name in class_colors:
        return class_colors[class_name]
    if class_name in KITTI_COLORS:
        return KITTI_COLORS[class_name]
    # Hash-based color for unknown classes
    idx = hash(class_name) % len(DETECTION_PALETTE)
    return DETECTION_PALETTE[idx]


def get_color_for_index(idx: int) -> Tuple[int, int, int]:
    """Get color by index from palette."""
    return DETECTION_PALETTE[idx % len(DETECTION_PALETTE)]


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_2d_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[str]] = None,
    scores: Optional[np.ndarray] = None,
    colors: Optional[Union[List[Tuple], Dict[str, Tuple]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_labels: bool = True,
    show_scores: bool = True,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Draw 2D bounding boxes on image.

    Args:
        image: (H, W, 3) RGB image.
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes.
        labels: Optional list of N class labels.
        scores: Optional (N,) array of confidence scores.
        colors: Color specification:
            - None: auto-color by label or index
            - List[Tuple]: per-box colors
            - Dict[str, Tuple]: label-to-color mapping
        thickness: Box line thickness.
        font_scale: Label font scale.
        show_labels: Whether to show class labels.
        show_scores: Whether to show confidence scores.
        alpha: Fill transparency (0 = no fill, 1 = solid fill).

    Returns:
        Image with boxes drawn.
    """
    result = image.copy()
    n_boxes = len(boxes)

    if n_boxes == 0:
        return result

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])

        # Determine color
        if isinstance(colors, list) and i < len(colors):
            color = colors[i]
        elif isinstance(colors, dict) and labels and i < len(labels):
            color = colors.get(labels[i], get_color_for_index(i))
        elif labels and i < len(labels):
            color = get_color_for_class(labels[i])
        else:
            color = get_color_for_index(i)

        # Draw filled rectangle with transparency
        if alpha > 0:
            overlay = result.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Draw box outline
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label_parts = []
        if show_labels and labels and i < len(labels):
            label_parts.append(labels[i])
        if show_scores and scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label_text = " ".join(label_parts)
            draw_text(
                result,
                label_text,
                (x1, y1 - 5),
                color=color,
                font_scale=font_scale,
                background=True,
            )

    return result


def draw_3d_boxes(
    image: np.ndarray,
    corners_list: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Tuple], Dict[str, Tuple]]] = None,
    thickness: int = 2,
    draw_orientation: bool = True,
) -> np.ndarray:
    """
    Draw projected 3D bounding boxes on image.

    Args:
        image: (H, W, 3) RGB image.
        corners_list: List of (8, 2) projected corner arrays.
            Corner order: 0-3 bottom face, 4-7 top face.
        labels: Optional class labels.
        colors: Color specification.
        thickness: Line thickness.
        draw_orientation: Draw front face differently.

    Returns:
        Image with 3D boxes drawn.
    """
    result = image.copy()

    for i, corners in enumerate(corners_list):
        # Determine color
        if isinstance(colors, list) and i < len(colors):
            color = colors[i]
        elif isinstance(colors, dict) and labels and i < len(labels):
            color = colors.get(labels[i], get_color_for_index(i))
        elif labels and i < len(labels):
            color = get_color_for_class(labels[i])
        else:
            color = get_color_for_index(i)

        corners = corners.astype(np.int32)

        # Check if corners are within image
        h, w = image.shape[:2]
        if (corners[:, 0].max() < 0 or corners[:, 0].min() > w or
                corners[:, 1].max() < 0 or corners[:, 1].min() > h):
            continue

        # Draw bottom face (0-1-2-3) - darker
        bottom_color = tuple(int(c * 0.7) for c in color)
        for j in range(4):
            pt1 = tuple(corners[j])
            pt2 = tuple(corners[(j + 1) % 4])
            cv2.line(result, pt1, pt2, bottom_color, thickness)

        # Draw top face (4-5-6-7) - brighter
        for j in range(4):
            pt1 = tuple(corners[j + 4])
            pt2 = tuple(corners[(j + 1) % 4 + 4])
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw vertical edges (pillars)
        for j in range(4):
            pt1 = tuple(corners[j])
            pt2 = tuple(corners[j + 4])
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw front face with X (orientation indicator)
        if draw_orientation:
            # Front face is defined by corners 0, 1, 4, 5
            front_color = (255, 255, 255)  # White for orientation
            cv2.line(result, tuple(corners[0]), tuple(corners[5]), front_color, 1)
            cv2.line(result, tuple(corners[1]), tuple(corners[4]), front_color, 1)

    return result


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 1,
    background: bool = True,
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 3,
) -> np.ndarray:
    """
    Draw text on image with optional background.

    Args:
        image: Image to draw on (modified in place).
        text: Text string to draw.
        position: (x, y) position (bottom-left of text).
        color: Text color (RGB).
        font_scale: Font scale factor.
        thickness: Text thickness.
        background: Whether to draw background box.
        bg_color: Background color (auto-computed if None).
        padding: Background padding in pixels.

    Returns:
        Image with text drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Ensure text is within image bounds
    h, w = image.shape[:2]
    y = max(text_h + padding, min(y, h - padding))
    x = max(padding, min(x, w - text_w - padding))

    if background:
        # Compute background color (darker version of text color or custom)
        if bg_color is None:
            bg_color = tuple(int(c * 0.3) for c in color)

        # Draw background rectangle
        bg_pt1 = (x - padding, y - text_h - padding)
        bg_pt2 = (x + text_w + padding, y + baseline + padding)
        cv2.rectangle(image, bg_pt1, bg_pt2, bg_color, -1)

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def draw_legend(
    image: np.ndarray,
    labels: List[str],
    colors: Optional[Dict[str, Tuple]] = None,
    position: str = "top-right",
    font_scale: float = 0.5,
    box_size: int = 15,
    padding: int = 10,
) -> np.ndarray:
    """
    Draw a legend on the image.

    Args:
        image: Image to draw on.
        labels: List of class labels.
        colors: Label to color mapping.
        position: Legend position ('top-right', 'top-left', 'bottom-right', 'bottom-left').
        font_scale: Font scale.
        box_size: Color box size.
        padding: Padding from image edge.

    Returns:
        Image with legend drawn.
    """
    result = image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate legend size
    max_text_w = 0
    text_height = 0
    for label in labels:
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        max_text_w = max(max_text_w, tw)
        text_height = max(text_height, th)

    legend_w = box_size + padding + max_text_w + padding * 2
    legend_h = (text_height + padding) * len(labels) + padding

    # Determine position
    if "right" in position:
        x0 = w - legend_w - padding
    else:
        x0 = padding

    if "bottom" in position:
        y0 = h - legend_h - padding
    else:
        y0 = padding

    # Draw background
    cv2.rectangle(result, (x0, y0), (x0 + legend_w, y0 + legend_h), (40, 40, 40), -1)
    cv2.rectangle(result, (x0, y0), (x0 + legend_w, y0 + legend_h), (100, 100, 100), 1)

    # Draw entries
    for i, label in enumerate(labels):
        color = get_color_for_class(label, colors)
        y = y0 + padding + i * (text_height + padding)

        # Color box
        cv2.rectangle(
            result,
            (x0 + padding, y),
            (x0 + padding + box_size, y + box_size),
            color,
            -1,
        )

        # Label text
        cv2.putText(
            result,
            label,
            (x0 + padding + box_size + padding, y + box_size - 2),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a binary mask on image.

    Args:
        image: (H, W, 3) RGB image.
        mask: (H, W) binary mask.
        color: Mask color (RGB).
        alpha: Transparency (0-1).

    Returns:
        Image with mask overlay.
    """
    result = image.copy()

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend
    mask_region = mask > 0
    result[mask_region] = cv2.addWeighted(
        image[mask_region], 1 - alpha,
        colored_mask[mask_region], alpha,
        0,
    )

    return result


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
    create_dir: bool = True,
) -> bool:
    """
    Save image to file.

    Args:
        image: (H, W, 3) RGB image.
        path: Output file path.
        quality: JPEG quality (1-100).
        create_dir: Create parent directories if needed.

    Returns:
        True if successful.
    """
    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Set compression parameters
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]
    else:
        params = []

    return cv2.imwrite(str(path), image_bgr, params)


# =============================================================================
# ImageOverlay Class (Fluent Interface)
# =============================================================================

class ImageOverlay:
    """
    Fluent interface for building image overlays.

    Usage:
        result = (ImageOverlay(image)
            .draw_boxes(boxes, labels)
            .draw_text("Frame 001", (10, 30))
            .draw_legend(["Car", "Pedestrian"])
            .get())
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize with base image.

        Args:
            image: (H, W, 3) RGB image.
        """
        self.image = image.copy()

    def draw_boxes(
        self,
        boxes: np.ndarray,
        labels: Optional[List[str]] = None,
        scores: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "ImageOverlay":
        """Draw 2D bounding boxes."""
        self.image = draw_2d_boxes(self.image, boxes, labels, scores, **kwargs)
        return self

    def draw_3d_boxes(
        self,
        corners_list: List[np.ndarray],
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> "ImageOverlay":
        """Draw 3D bounding boxes."""
        self.image = draw_3d_boxes(self.image, corners_list, labels, **kwargs)
        return self

    def draw_text(
        self,
        text: str,
        position: Tuple[int, int],
        **kwargs,
    ) -> "ImageOverlay":
        """Draw text."""
        self.image = draw_text(self.image, text, position, **kwargs)
        return self

    def draw_legend(
        self,
        labels: List[str],
        **kwargs,
    ) -> "ImageOverlay":
        """Draw legend."""
        self.image = draw_legend(self.image, labels, **kwargs)
        return self

    def overlay_mask(
        self,
        mask: np.ndarray,
        **kwargs,
    ) -> "ImageOverlay":
        """Overlay binary mask."""
        self.image = overlay_mask(self.image, mask, **kwargs)
        return self

    def get(self) -> np.ndarray:
        """Get the final image."""
        return self.image

    def save(self, path: Union[str, Path], **kwargs) -> bool:
        """Save image to file."""
        return save_image(self.image, path, **kwargs)
