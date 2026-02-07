"""
Bird's Eye View (BEV) Visualization for Multi-Sensor Perception.

Professional-grade visualization inspired by Waymo/Tesla perception displays.
Provides top-down view of LiDAR points, 3D bounding boxes, and ego vehicle.

Features:
- High-quality BEV rendering with configurable range
- LiDAR point cloud visualization with height/intensity coloring
- 3D bounding box rendering with orientation arrows
- Ego vehicle and camera frustum visualization
- Distance grid and range markers
- Multi-panel display combining camera and BEV views
- Real-time metrics overlay

Author: Multi-Sensor Perception Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import numpy as np
import cv2

# Try to import matplotlib for colormaps
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration Classes
# =============================================================================

class ColorScheme(Enum):
    """Color schemes for BEV visualization."""
    DARK = "dark"       # Dark background (Waymo style)
    LIGHT = "light"     # Light background
    MIDNIGHT = "midnight"  # Deep blue background (Tesla style)


@dataclass
class BEVConfig:
    """
    Configuration for Bird's Eye View visualization.

    Attributes:
        x_range: Range in X direction (left-right) in meters.
        y_range: Range in Y direction (forward) in meters.
        resolution: Pixels per meter.
        background_color: Background color (BGR).
        grid_color: Grid line color (BGR).
        grid_spacing: Distance between grid lines in meters.
        show_grid: Whether to show distance grid.
        show_range_circles: Whether to show range circles.
        show_ego_vehicle: Whether to show ego vehicle marker.
        show_camera_fov: Whether to show camera field of view.
        point_size: Size of LiDAR points.
        box_thickness: Thickness of bounding box lines.
        arrow_length: Length of orientation arrows.
        color_scheme: Overall color scheme.
    """
    x_range: Tuple[float, float] = (-25.0, 25.0)
    y_range: Tuple[float, float] = (0.0, 50.0)
    resolution: float = 10.0  # pixels per meter
    background_color: Tuple[int, int, int] = (20, 20, 25)
    grid_color: Tuple[int, int, int] = (50, 50, 55)
    grid_spacing: float = 10.0
    show_grid: bool = True
    show_range_circles: bool = True
    show_ego_vehicle: bool = True
    show_camera_fov: bool = True
    point_size: int = 2
    box_thickness: int = 2
    arrow_length: float = 2.0
    color_scheme: ColorScheme = ColorScheme.DARK

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return int((self.x_range[1] - self.x_range[0]) * self.resolution)

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return int((self.y_range[1] - self.y_range[0]) * self.resolution)


# =============================================================================
# Color Palettes
# =============================================================================

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    "Car": (0, 255, 100),           # Bright green
    "Van": (0, 200, 150),           # Teal
    "Truck": (0, 150, 255),         # Orange
    "Pedestrian": (100, 100, 255),   # Red-ish
    "Person_sitting": (150, 100, 255),
    "Cyclist": (0, 255, 255),       # Yellow
    "Tram": (255, 100, 100),        # Blue
    "Misc": (150, 150, 150),        # Gray
    "Unknown": (128, 128, 128),
}

# Depth colormap (near=warm, far=cool)
def get_depth_color(depth: float, min_depth: float = 0.0, max_depth: float = 50.0) -> Tuple[int, int, int]:
    """Get color based on depth using a perceptually uniform colormap."""
    normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)

    if HAS_MATPLOTLIB:
        # Use matplotlib's plasma colormap
        rgba = cm.plasma(1 - normalized)
        return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
    else:
        # Fallback: simple blue-to-red gradient
        r = int(255 * normalized)
        b = int(255 * (1 - normalized))
        g = int(100 * (1 - abs(normalized - 0.5) * 2))
        return (b, g, r)


def get_height_color(height: float, min_height: float = -2.0, max_height: float = 2.0) -> Tuple[int, int, int]:
    """Get color based on height (Z in LiDAR frame, Y in camera frame)."""
    normalized = np.clip((height - min_height) / (max_height - min_height), 0, 1)

    if HAS_MATPLOTLIB:
        rgba = cm.viridis(normalized)
        return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
    else:
        # Fallback gradient
        r = int(50 + 200 * normalized)
        g = int(255 * (1 - abs(normalized - 0.5)))
        b = int(255 * (1 - normalized))
        return (b, g, r)


# =============================================================================
# Main BEV Visualizer Class
# =============================================================================

class BEVVisualizer:
    """
    Bird's Eye View visualizer for autonomous vehicle perception.

    Creates professional-grade top-down visualizations of:
    - LiDAR point clouds
    - 3D bounding boxes with orientation
    - Ego vehicle position
    - Distance grid and range markers

    Example:
        >>> config = BEVConfig(x_range=(-30, 30), y_range=(0, 60))
        >>> bev = BEVVisualizer(config)
        >>> canvas = bev.render(points=lidar_points, boxes=boxes_3d)
    """

    def __init__(self, config: Optional[BEVConfig] = None):
        """
        Initialize BEV visualizer.

        Args:
            config: BEV configuration. Uses defaults if None.
        """
        self.config = config or BEVConfig()
        self._setup_color_scheme()

    def _setup_color_scheme(self):
        """Setup colors based on selected scheme."""
        scheme = self.config.color_scheme

        if scheme == ColorScheme.DARK:
            self.bg_color = (20, 20, 25)
            self.grid_color = (50, 50, 55)
            self.text_color = (200, 200, 200)
            self.ego_color = (100, 255, 100)
            self.fov_color = (80, 80, 100)
        elif scheme == ColorScheme.LIGHT:
            self.bg_color = (240, 240, 245)
            self.grid_color = (200, 200, 210)
            self.text_color = (50, 50, 50)
            self.ego_color = (0, 150, 0)
            self.fov_color = (200, 200, 220)
        elif scheme == ColorScheme.MIDNIGHT:
            self.bg_color = (15, 15, 35)
            self.grid_color = (40, 40, 70)
            self.text_color = (180, 180, 220)
            self.ego_color = (50, 255, 150)
            self.fov_color = (60, 60, 100)

    # -------------------------------------------------------------------------
    # Coordinate Transformation
    # -------------------------------------------------------------------------

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to pixel coordinates.

        In BEV: X is left-right, Y is forward (depth).
        Origin is at bottom center of image.

        Args:
            x: X coordinate in meters (positive = right).
            y: Y coordinate in meters (positive = forward).

        Returns:
            Pixel coordinates (u, v).
        """
        # X: center of image, positive right
        u = int((x - self.config.x_range[0]) * self.config.resolution)
        # Y: bottom of image is y_range[0], top is y_range[1]
        v = int((self.config.y_range[1] - y) * self.config.resolution)
        return (u, v)

    def pixel_to_world(self, u: int, v: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        x = u / self.config.resolution + self.config.x_range[0]
        y = self.config.y_range[1] - v / self.config.resolution
        return (x, y)

    # -------------------------------------------------------------------------
    # Canvas Creation
    # -------------------------------------------------------------------------

    def create_canvas(self) -> np.ndarray:
        """Create empty BEV canvas with background."""
        canvas = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        return canvas

    # -------------------------------------------------------------------------
    # Drawing Methods
    # -------------------------------------------------------------------------

    def draw_grid(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw distance grid on canvas.

        Args:
            canvas: BEV canvas to draw on.

        Returns:
            Canvas with grid.
        """
        spacing = self.config.grid_spacing

        # Vertical lines (constant X)
        x = self.config.x_range[0]
        while x <= self.config.x_range[1]:
            u, _ = self.world_to_pixel(x, 0)
            cv2.line(canvas, (u, 0), (u, self.config.height), self.grid_color, 1)
            x += spacing

        # Horizontal lines (constant Y)
        y = self.config.y_range[0]
        while y <= self.config.y_range[1]:
            _, v = self.world_to_pixel(0, y)
            cv2.line(canvas, (0, v), (self.config.width, v), self.grid_color, 1)

            # Add distance label
            if y > 0:
                label_u, label_v = self.world_to_pixel(self.config.x_range[0] + 1, y)
                cv2.putText(canvas, f"{int(y)}m", (label_u, label_v + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
            y += spacing

        return canvas

    def draw_range_circles(self, canvas: np.ndarray, radii: List[float] = None) -> np.ndarray:
        """
        Draw range circles centered on ego vehicle.

        Args:
            canvas: BEV canvas.
            radii: List of circle radii in meters.

        Returns:
            Canvas with range circles.
        """
        if radii is None:
            radii = [10, 20, 30, 40, 50]

        center = self.world_to_pixel(0, 0)

        for r in radii:
            radius_px = int(r * self.config.resolution)
            cv2.circle(canvas, center, radius_px, self.grid_color, 1)

            # Add label
            label_pos = self.world_to_pixel(r * 0.7, r * 0.7)
            cv2.putText(canvas, f"{int(r)}m", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1)

        return canvas

    def draw_ego_vehicle(self, canvas: np.ndarray, length: float = 4.5, width: float = 2.0) -> np.ndarray:
        """
        Draw ego vehicle marker at origin.

        Args:
            canvas: BEV canvas.
            length: Vehicle length in meters.
            width: Vehicle width in meters.

        Returns:
            Canvas with ego vehicle.
        """
        # Vehicle corners (rear-left, rear-right, front-right, front-left)
        corners = np.array([
            [-width/2, -length/4],
            [width/2, -length/4],
            [width/2, length * 3/4],
            [-width/2, length * 3/4],
        ])

        # Convert to pixels
        pts = np.array([self.world_to_pixel(c[0], c[1]) for c in corners], dtype=np.int32)

        # Draw filled vehicle shape
        cv2.fillPoly(canvas, [pts], self.ego_color)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)

        # Draw direction arrow
        arrow_start = self.world_to_pixel(0, length/2)
        arrow_end = self.world_to_pixel(0, length)
        cv2.arrowedLine(canvas, arrow_start, arrow_end, (255, 255, 255), 2, tipLength=0.4)

        return canvas

    def draw_camera_fov(
        self,
        canvas: np.ndarray,
        fov_degrees: float = 90.0,
        max_range: float = 50.0,
    ) -> np.ndarray:
        """
        Draw camera field of view cone.

        Args:
            canvas: BEV canvas.
            fov_degrees: Camera horizontal FOV in degrees.
            max_range: Maximum visualization range.

        Returns:
            Canvas with FOV cone.
        """
        half_fov = np.radians(fov_degrees / 2)

        # FOV boundary points
        left_x = -max_range * np.tan(half_fov)
        right_x = max_range * np.tan(half_fov)

        origin = self.world_to_pixel(0, 0)
        left_pt = self.world_to_pixel(left_x, max_range)
        right_pt = self.world_to_pixel(right_x, max_range)

        # Draw FOV cone (semi-transparent)
        overlay = canvas.copy()
        pts = np.array([origin, left_pt, right_pt], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], self.fov_color)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw FOV boundary lines
        cv2.line(canvas, origin, left_pt, self.fov_color, 1)
        cv2.line(canvas, origin, right_pt, self.fov_color, 1)

        return canvas

    def draw_lidar_points(
        self,
        canvas: np.ndarray,
        points: np.ndarray,
        color_by: str = "height",
        intensity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw LiDAR points on BEV canvas.

        Args:
            canvas: BEV canvas.
            points: LiDAR points (N, 3) in camera frame [X, Y, Z].
                   X=right, Y=down, Z=forward
            color_by: Coloring method - "height", "depth", "intensity", or "uniform".
            intensity: Optional intensity values (N,) for intensity coloring.

        Returns:
            Canvas with LiDAR points.
        """
        if len(points) == 0:
            return canvas

        # Convert from camera frame to BEV coordinates
        # Camera: X=right, Y=down, Z=forward
        # BEV: X=right, Y=forward
        bev_x = points[:, 0]  # Right
        bev_y = points[:, 2]  # Forward (depth)
        heights = -points[:, 1]  # Up (negative Y in camera frame)

        # Filter points within range
        mask = (
            (bev_x >= self.config.x_range[0]) & (bev_x <= self.config.x_range[1]) &
            (bev_y >= self.config.y_range[0]) & (bev_y <= self.config.y_range[1])
        )

        bev_x = bev_x[mask]
        bev_y = bev_y[mask]
        heights = heights[mask]

        if intensity is not None:
            intensity = intensity[mask]

        # Get colors
        for i in range(len(bev_x)):
            if color_by == "height":
                color = get_height_color(heights[i])
            elif color_by == "depth":
                color = get_depth_color(bev_y[i], self.config.y_range[0], self.config.y_range[1])
            elif color_by == "intensity" and intensity is not None:
                norm_int = np.clip(intensity[i] / 255.0, 0, 1)
                gray = int(50 + 200 * norm_int)
                color = (gray, gray, gray)
            else:
                color = (150, 150, 150)

            u, v = self.world_to_pixel(bev_x[i], bev_y[i])
            cv2.circle(canvas, (u, v), self.config.point_size, color, -1)

        return canvas

    def draw_3d_box(
        self,
        canvas: np.ndarray,
        center: np.ndarray,
        dimensions: np.ndarray,
        rotation_y: float,
        class_name: str = "Unknown",
        score: float = 1.0,
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Draw a single 3D bounding box on BEV.

        Args:
            canvas: BEV canvas.
            center: Box center (x, y, z) in camera frame.
            dimensions: Box dimensions (length, width, height).
            rotation_y: Rotation around Y-axis in radians.
            class_name: Object class name.
            score: Detection confidence score.
            show_label: Whether to show class label.

        Returns:
            Canvas with bounding box.
        """
        # Get color for class
        color = CLASS_COLORS.get(class_name, CLASS_COLORS["Unknown"])

        # Box center in BEV coordinates
        cx = center[0]  # X (right)
        cy = center[2]  # Z (forward/depth)

        # Box dimensions
        length = dimensions[0]  # Length (along forward direction)
        width = dimensions[1]   # Width (left-right)

        # Compute corners in local frame
        # Note: rotation_y rotates around Y (up in camera, into page in BEV)
        half_l = length / 2
        half_w = width / 2

        # Local corners (before rotation)
        local_corners = np.array([
            [half_l, half_w],    # Front-right
            [half_l, -half_w],   # Front-left
            [-half_l, -half_w],  # Rear-left
            [-half_l, half_w],   # Rear-right
        ])

        # Rotation matrix (around Y-axis, which is "up" in BEV)
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, -s], [s, c]])

        # Rotate and translate corners
        world_corners = (local_corners @ R.T) + np.array([cx, cy])

        # Convert to pixels
        pixel_corners = np.array([
            self.world_to_pixel(corner[0], corner[1])
            for corner in world_corners
        ], dtype=np.int32)

        # Draw filled box (semi-transparent)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pixel_corners], color)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Draw box outline
        cv2.polylines(canvas, [pixel_corners], True, color, self.config.box_thickness)

        # Draw front edge (thicker)
        front_edge = pixel_corners[:2]
        cv2.line(canvas, tuple(front_edge[0]), tuple(front_edge[1]),
                (255, 255, 255), self.config.box_thickness + 1)

        # Draw orientation arrow
        arrow_start = self.world_to_pixel(cx, cy)
        arrow_dx = self.config.arrow_length * np.sin(rotation_y)
        arrow_dy = self.config.arrow_length * np.cos(rotation_y)
        arrow_end = self.world_to_pixel(cx + arrow_dx, cy + arrow_dy)
        cv2.arrowedLine(canvas, arrow_start, arrow_end, (255, 255, 255), 2, tipLength=0.3)

        # Draw label
        if show_label:
            label_pos = self.world_to_pixel(cx, cy + length/2 + 1)
            label = f"{class_name}"
            if score < 1.0:
                label += f" {score:.2f}"
            cv2.putText(canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

        return canvas

    def draw_3d_boxes(
        self,
        canvas: np.ndarray,
        boxes: List[Any],  # List of BBox3D objects
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw multiple 3D bounding boxes.

        Args:
            canvas: BEV canvas.
            boxes: List of BBox3D objects.
            show_labels: Whether to show class labels.

        Returns:
            Canvas with all boxes.
        """
        for box in boxes:
            if box is None:
                continue

            canvas = self.draw_3d_box(
                canvas,
                center=box.center,
                dimensions=box.dimensions,
                rotation_y=box.rotation_y,
                class_name=box.class_name,
                score=box.score,
                show_label=show_labels,
            )

        return canvas

    def add_legend(self, canvas: np.ndarray, classes: List[str] = None) -> np.ndarray:
        """
        Add class color legend to canvas.

        Args:
            canvas: BEV canvas.
            classes: List of classes to show. Shows all if None.

        Returns:
            Canvas with legend.
        """
        if classes is None:
            classes = ["Car", "Pedestrian", "Cyclist", "Truck"]

        # Legend position (top-right)
        x_start = self.config.width - 100
        y_start = 20
        line_height = 20

        # Background
        cv2.rectangle(canvas, (x_start - 10, y_start - 10),
                     (self.config.width - 5, y_start + len(classes) * line_height + 5),
                     (40, 40, 45), -1)

        for i, cls in enumerate(classes):
            color = CLASS_COLORS.get(cls, (128, 128, 128))
            y = y_start + i * line_height

            # Color box
            cv2.rectangle(canvas, (x_start, y), (x_start + 15, y + 12), color, -1)

            # Label
            cv2.putText(canvas, cls, (x_start + 20, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1)

        return canvas

    def add_title(self, canvas: np.ndarray, title: str) -> np.ndarray:
        """Add title to canvas."""
        cv2.putText(canvas, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        return canvas

    # -------------------------------------------------------------------------
    # Main Render Method
    # -------------------------------------------------------------------------

    def render(
        self,
        points: Optional[np.ndarray] = None,
        boxes: Optional[List[Any]] = None,
        title: str = "Bird's Eye View",
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Render complete BEV visualization.

        Args:
            points: LiDAR points (N, 3) in camera frame.
            boxes: List of BBox3D objects.
            title: Title text.
            show_legend: Whether to show class legend.

        Returns:
            Complete BEV visualization image.
        """
        # Create canvas
        canvas = self.create_canvas()

        # Draw layers (back to front)
        if self.config.show_camera_fov:
            canvas = self.draw_camera_fov(canvas)

        if self.config.show_grid:
            canvas = self.draw_grid(canvas)

        if self.config.show_range_circles:
            canvas = self.draw_range_circles(canvas)

        # Draw LiDAR points
        if points is not None:
            canvas = self.draw_lidar_points(canvas, points, color_by="height")

        # Draw boxes
        if boxes is not None:
            canvas = self.draw_3d_boxes(canvas, boxes)

        # Draw ego vehicle (on top)
        if self.config.show_ego_vehicle:
            canvas = self.draw_ego_vehicle(canvas)

        # Add legend and title
        if show_legend:
            canvas = self.add_legend(canvas)

        canvas = self.add_title(canvas, title)

        return canvas


# =============================================================================
# Multi-Panel Display
# =============================================================================

class MultiPanelDisplay:
    """
    Multi-panel visualization combining camera view, BEV, and metrics.

    Layout:
        +------------------+------------------+
        |   Camera View    |    BEV View     |
        | (2D + 3D boxes)  | (LiDAR + boxes) |
        +------------------+------------------+
        |           Metrics Panel            |
        +------------------------------------+
    """

    def __init__(
        self,
        camera_size: Tuple[int, int] = (800, 375),
        bev_size: Tuple[int, int] = (500, 500),
        metrics_height: int = 80,
        padding: int = 5,
        background_color: Tuple[int, int, int] = (30, 30, 35),
    ):
        """
        Initialize multi-panel display.

        Args:
            camera_size: Size of camera panel (width, height).
            bev_size: Size of BEV panel (width, height).
            metrics_height: Height of metrics panel.
            padding: Padding between panels.
            background_color: Background color.
        """
        self.camera_size = camera_size
        self.bev_size = bev_size
        self.metrics_height = metrics_height
        self.padding = padding
        self.bg_color = background_color

        # Calculate total size
        top_height = max(camera_size[1], bev_size[1])
        self.total_width = camera_size[0] + bev_size[0] + 3 * padding
        self.total_height = top_height + metrics_height + 3 * padding

        # Panel positions
        self.camera_pos = (padding, padding)
        self.bev_pos = (camera_size[0] + 2 * padding, padding)
        self.metrics_pos = (padding, top_height + 2 * padding)

    def create_canvas(self) -> np.ndarray:
        """Create empty multi-panel canvas."""
        canvas = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        return canvas

    def add_camera_panel(
        self,
        canvas: np.ndarray,
        camera_image: np.ndarray,
        title: str = "Camera View",
    ) -> np.ndarray:
        """Add camera view panel."""
        # Resize camera image
        resized = cv2.resize(camera_image, self.camera_size)

        # Add title
        cv2.putText(resized, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Place on canvas
        x, y = self.camera_pos
        canvas[y:y+self.camera_size[1], x:x+self.camera_size[0]] = resized

        # Draw border
        cv2.rectangle(canvas, (x-1, y-1),
                     (x+self.camera_size[0], y+self.camera_size[1]),
                     (60, 60, 65), 1)

        return canvas

    def add_bev_panel(
        self,
        canvas: np.ndarray,
        bev_image: np.ndarray,
        title: str = "Bird's Eye View",
    ) -> np.ndarray:
        """Add BEV panel."""
        # Resize BEV image
        resized = cv2.resize(bev_image, self.bev_size)

        # Place on canvas
        x, y = self.bev_pos
        canvas[y:y+self.bev_size[1], x:x+self.bev_size[0]] = resized

        # Draw border
        cv2.rectangle(canvas, (x-1, y-1),
                     (x+self.bev_size[0], y+self.bev_size[1]),
                     (60, 60, 65), 1)

        return canvas

    def add_metrics_panel(
        self,
        canvas: np.ndarray,
        metrics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Add metrics panel at bottom.

        Args:
            canvas: Multi-panel canvas.
            metrics: Dictionary of metrics to display.
                Expected keys: detections, avg_depth, fps, frame_id, etc.
        """
        x, y = self.metrics_pos
        panel_width = self.total_width - 2 * self.padding

        # Background
        cv2.rectangle(canvas, (x, y), (x + panel_width, y + self.metrics_height),
                     (40, 40, 45), -1)
        cv2.rectangle(canvas, (x, y), (x + panel_width, y + self.metrics_height),
                     (60, 60, 65), 1)

        # Metrics text
        text_y = y + 25
        text_x = x + 20
        col_width = panel_width // 5

        # Frame info
        if "frame_id" in metrics:
            cv2.putText(canvas, f"Frame: {metrics['frame_id']}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Detections
        text_x += col_width
        if "detections" in metrics:
            det_count = metrics["detections"]
            cv2.putText(canvas, f"Detections: {det_count}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # 3D Boxes
        text_x += col_width
        if "boxes_3d" in metrics:
            box_count = metrics["boxes_3d"]
            cv2.putText(canvas, f"3D Boxes: {box_count}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Average depth
        text_x += col_width
        if "avg_depth" in metrics:
            avg_depth = metrics["avg_depth"]
            cv2.putText(canvas, f"Avg Depth: {avg_depth:.1f}m", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

        # FPS
        text_x += col_width
        if "fps" in metrics:
            fps = metrics["fps"]
            color = (100, 255, 100) if fps > 10 else (100, 100, 255)
            cv2.putText(canvas, f"FPS: {fps:.1f}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Second row - class breakdown
        text_y = y + 55
        text_x = x + 20

        if "class_counts" in metrics:
            for cls, count in metrics["class_counts"].items():
                color = CLASS_COLORS.get(cls, (150, 150, 150))
                cv2.putText(canvas, f"{cls}: {count}", (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                text_x += 100

        return canvas

    def render(
        self,
        camera_image: np.ndarray,
        bev_image: np.ndarray,
        metrics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Render complete multi-panel display.

        Args:
            camera_image: Camera view image.
            bev_image: BEV visualization image.
            metrics: Metrics dictionary.

        Returns:
            Complete multi-panel visualization.
        """
        canvas = self.create_canvas()
        canvas = self.add_camera_panel(canvas, camera_image)
        canvas = self.add_bev_panel(canvas, bev_image)
        canvas = self.add_metrics_panel(canvas, metrics)
        return canvas


# =============================================================================
# Utility Functions
# =============================================================================

def create_comparison_view(
    image_before: np.ndarray,
    image_after: np.ndarray,
    title_before: str = "Before Fusion",
    title_after: str = "After Fusion",
) -> np.ndarray:
    """
    Create side-by-side comparison view.

    Args:
        image_before: Image before processing.
        image_after: Image after processing.
        title_before: Title for left panel.
        title_after: Title for right panel.

    Returns:
        Combined comparison image.
    """
    # Ensure same height
    h1, w1 = image_before.shape[:2]
    h2, w2 = image_after.shape[:2]

    target_h = max(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        image_before = cv2.resize(image_before, (int(w1 * scale), target_h))

    if h2 != target_h:
        scale = target_h / h2
        image_after = cv2.resize(image_after, (int(w2 * scale), target_h))

    # Add titles
    cv2.putText(image_before, title_before, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image_after, title_after, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine with separator
    separator = np.zeros((target_h, 5, 3), dtype=np.uint8)
    separator[:] = (100, 100, 100)

    return np.hstack([image_before, separator, image_after])


def draw_depth_improvement_indicator(
    image: np.ndarray,
    detections_2d: int,
    detections_with_depth: int,
    position: Tuple[int, int] = (10, 50),
) -> np.ndarray:
    """
    Draw indicator showing how many detections got valid depth.

    Args:
        image: Input image.
        detections_2d: Number of 2D detections.
        detections_with_depth: Number of detections with valid depth.
        position: Position for indicator.

    Returns:
        Image with indicator.
    """
    result = image.copy()

    if detections_2d > 0:
        ratio = detections_with_depth / detections_2d
        percentage = int(ratio * 100)

        # Color based on success rate
        if ratio >= 0.9:
            color = (100, 255, 100)  # Green
        elif ratio >= 0.7:
            color = (100, 200, 255)  # Orange
        else:
            color = (100, 100, 255)  # Red

        text = f"Depth Coverage: {percentage}% ({detections_with_depth}/{detections_2d})"
    else:
        color = (150, 150, 150)
        text = "No detections"

    cv2.putText(result, text, position,
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result
