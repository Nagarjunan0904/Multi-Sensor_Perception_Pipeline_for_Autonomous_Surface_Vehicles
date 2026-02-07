"""
Point cloud visualization utilities.

Production-quality 3D visualization using Open3D and 2D projections
for LiDAR-camera fusion visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Open3D is optional for headless environments
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# =============================================================================
# Colormaps
# =============================================================================

def depth_to_color(
    depth: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 70.0,
    colormap: str = "turbo",
) -> np.ndarray:
    """
    Convert depth values to RGB colors using colormap.

    Args:
        depth: (N,) array of depth values.
        min_depth: Minimum depth for normalization.
        max_depth: Maximum depth for normalization.
        colormap: Colormap name ('turbo', 'jet', 'viridis', 'plasma').

    Returns:
        colors: (N, 3) array of RGB colors [0-255].
    """
    # Normalize depth to [0, 1]
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)

    # Convert to uint8 for colormap lookup
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # Apply colormap
    colormap_cv = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT,
        "rainbow": cv2.COLORMAP_RAINBOW,
    }.get(colormap, cv2.COLORMAP_TURBO)

    # Apply colormap (returns BGR)
    colored = cv2.applyColorMap(depth_uint8.reshape(-1, 1), colormap_cv)
    colored = colored.reshape(-1, 3)

    # Convert BGR to RGB
    colors = colored[:, ::-1]

    return colors


def intensity_to_color(
    intensity: np.ndarray,
    colormap: str = "viridis",
) -> np.ndarray:
    """
    Convert intensity values to RGB colors.

    Args:
        intensity: (N,) array of intensity values.
        colormap: Colormap name.

    Returns:
        colors: (N, 3) array of RGB colors [0-255].
    """
    # Normalize intensity
    if intensity.max() > 1.0:
        intensity = intensity / 255.0
    intensity = np.clip(intensity, 0, 1)

    return depth_to_color(intensity, 0, 1, colormap)


def height_to_color(
    height: np.ndarray,
    min_height: float = -3.0,
    max_height: float = 3.0,
    colormap: str = "rainbow",
) -> np.ndarray:
    """
    Convert height (z) values to RGB colors.

    Args:
        height: (N,) array of height values.
        min_height: Minimum height for normalization.
        max_height: Maximum height for normalization.
        colormap: Colormap name.

    Returns:
        colors: (N, 3) array of RGB colors [0-255].
    """
    return depth_to_color(height, min_height, max_height, colormap)


# =============================================================================
# 2D Projection Functions
# =============================================================================

def project_points_to_image(
    image: np.ndarray,
    points: np.ndarray,
    calib: Any,
    color_mode: str = "depth",
    point_size: int = 2,
    min_depth: float = 1.0,
    max_depth: float = 70.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Project LiDAR points onto camera image with depth coloring.

    This creates the classic "colored dots" LiDAR-camera overlay visualization.

    Args:
        image: (H, W, 3) RGB image.
        points: (N, 4) point cloud [x, y, z, intensity] in LiDAR frame.
        calib: Calibration object with project_velo_to_image() and get_fov_mask().
        color_mode: Coloring mode:
            - 'depth': Color by distance (blue=near, red=far)
            - 'intensity': Color by reflectance
            - 'height': Color by z coordinate
        point_size: Size of projected points in pixels.
        min_depth: Minimum depth for colormap (meters).
        max_depth: Maximum depth for colormap (meters).
        alpha: Overlay opacity (0-1).

    Returns:
        overlay: Image with projected point cloud.
    """
    h, w = image.shape[:2]

    # Get points in camera FOV
    fov_mask = calib.get_fov_mask(points, image.shape)
    points_fov = points[fov_mask]

    if len(points_fov) == 0:
        return image.copy()

    # Project to image coordinates
    pts_2d = calib.project_velo_to_image(points_fov)

    # Get depth in camera frame
    pts_rect = calib.project_velo_to_rect(points_fov)
    depth = pts_rect[:, 2]  # z in camera coordinates

    # Compute colors based on mode
    if color_mode == "depth":
        colors = depth_to_color(depth, min_depth, max_depth)
    elif color_mode == "intensity":
        colors = intensity_to_color(points_fov[:, 3])
    elif color_mode == "height":
        colors = height_to_color(points_fov[:, 2])  # z in LiDAR frame
    else:
        # Default: uniform color
        colors = np.full((len(points_fov), 3), [0, 255, 127], dtype=np.uint8)

    # Create overlay
    if alpha < 1.0:
        overlay = image.copy()
        points_layer = image.copy()
    else:
        overlay = image.copy()
        points_layer = overlay

    # Sort by depth (draw far points first, near points on top)
    depth_order = np.argsort(-depth)

    # Draw points
    for idx in depth_order:
        u, v = int(pts_2d[idx, 0]), int(pts_2d[idx, 1])
        if 0 <= u < w and 0 <= v < h:
            color = tuple(map(int, colors[idx]))
            cv2.circle(points_layer, (u, v), point_size, color, -1)

    # Blend if alpha < 1
    if alpha < 1.0:
        # Only blend where points were drawn
        mask = np.any(points_layer != image, axis=2)
        overlay[mask] = cv2.addWeighted(
            image[mask], 1 - alpha,
            points_layer[mask], alpha,
            0,
        )
    else:
        overlay = points_layer

    return overlay


def save_pointcloud_image(
    points: np.ndarray,
    path: Union[str, Path],
    image_size: Tuple[int, int] = (800, 600),
    view: str = "bev",
    color_mode: str = "height",
    point_size: int = 1,
    background_color: Tuple[int, int, int] = (30, 30, 30),
) -> bool:
    """
    Save point cloud as 2D image (BEV or side view).

    Args:
        points: (N, 4) point cloud [x, y, z, intensity].
        path: Output file path.
        image_size: (width, height) of output image.
        view: View type ('bev' for bird's eye, 'side' for side view).
        color_mode: Coloring mode ('height', 'intensity', 'depth').
        point_size: Size of points.
        background_color: Background color (RGB).

    Returns:
        True if successful.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    w, h = image_size

    # Create background
    image = np.full((h, w, 3), background_color, dtype=np.uint8)

    if len(points) == 0:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return True

    # Select view coordinates
    if view == "bev":
        # Bird's eye view: x = forward, y = left
        x_coords = points[:, 0]  # forward
        y_coords = points[:, 1]  # left
        # Range for BEV (typical driving scenario)
        x_range = (0, 70)  # 0-70m forward
        y_range = (-40, 40)  # 40m left/right
    elif view == "side":
        # Side view: x = forward, z = up
        x_coords = points[:, 0]  # forward
        y_coords = -points[:, 2]  # up (negate for image coords)
        x_range = (0, 70)
        y_range = (-3, 5)  # -3 to 5m height
    else:
        raise ValueError(f"Unknown view: {view}")

    # Compute colors
    if color_mode == "height":
        colors = height_to_color(points[:, 2])
    elif color_mode == "intensity":
        colors = intensity_to_color(points[:, 3])
    elif color_mode == "depth":
        depth = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        colors = depth_to_color(depth, 0, 70)
    else:
        colors = np.full((len(points), 3), [0, 255, 127], dtype=np.uint8)

    # Map coordinates to image
    x_scale = w / (x_range[1] - x_range[0])
    y_scale = h / (y_range[1] - y_range[0])

    u = ((x_coords - x_range[0]) * x_scale).astype(int)
    v = ((y_coords - y_range[0]) * y_scale).astype(int)

    # Filter points within image bounds
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, colors = u[valid], v[valid], colors[valid]

    # Draw points (sort by depth for proper occlusion)
    if view == "bev":
        order = np.argsort(-points[valid, 0])  # far to near
    else:
        order = np.arange(len(u))

    for idx in order:
        color = tuple(map(int, colors[idx]))
        cv2.circle(image, (u[idx], v[idx]), point_size, color, -1)

    # Save
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return True


def create_bev_image(
    points: np.ndarray,
    x_range: Tuple[float, float] = (0, 70.0),
    y_range: Tuple[float, float] = (-40, 40),
    z_range: Tuple[float, float] = (-3, 1),
    resolution: float = 0.1,
    color_mode: str = "height",
) -> np.ndarray:
    """
    Create bird's eye view image from point cloud.

    Args:
        points: (N, 4) point cloud [x, y, z, intensity].
        x_range: (min, max) range in x (forward) direction.
        y_range: (min, max) range in y (left) direction.
        z_range: (min, max) range in z (up) direction for filtering.
        resolution: Meters per pixel.
        color_mode: Coloring mode ('height', 'intensity', 'density').

    Returns:
        bev_image: (H, W, 3) RGB bird's eye view image.
    """
    # Filter points within range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    )
    points_filtered = points[mask]

    # Compute image size
    width = int((y_range[1] - y_range[0]) / resolution)
    height = int((x_range[1] - x_range[0]) / resolution)

    # Initialize image
    bev = np.zeros((height, width, 3), dtype=np.uint8)

    if len(points_filtered) == 0:
        return bev

    # Map points to pixels
    px = ((points_filtered[:, 0] - x_range[0]) / resolution).astype(int)
    py = ((points_filtered[:, 1] - y_range[0]) / resolution).astype(int)

    # Flip x for image coordinates (forward = up in image)
    px = height - 1 - px

    # Clamp to valid range
    px = np.clip(px, 0, height - 1)
    py = np.clip(py, 0, width - 1)

    # Compute colors
    if color_mode == "height":
        colors = height_to_color(points_filtered[:, 2], z_range[0], z_range[1])
    elif color_mode == "intensity":
        colors = intensity_to_color(points_filtered[:, 3])
    elif color_mode == "density":
        # Use uniform color, density shown by accumulation
        colors = np.full((len(points_filtered), 3), [0, 255, 127], dtype=np.uint8)
    else:
        colors = np.full((len(points_filtered), 3), [255, 255, 255], dtype=np.uint8)

    # Draw points (take max color for overlapping points)
    for i in range(len(px)):
        current = bev[px[i], py[i]]
        new_color = colors[i]
        # Use brighter color (simple max)
        bev[px[i], py[i]] = np.maximum(current, new_color)

    return bev


# =============================================================================
# Open3D 3D Visualization
# =============================================================================

def visualize_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    color_mode: str = "height",
    window_name: str = "Point Cloud",
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    show_axes: bool = True,
    boxes: Optional[List[np.ndarray]] = None,
    box_colors: Optional[List[Tuple]] = None,
) -> None:
    """
    Visualize point cloud in 3D using Open3D.

    Args:
        points: (N, 4) point cloud [x, y, z, intensity].
        colors: Optional (N, 3) RGB colors [0-1]. If None, auto-colored.
        color_mode: Auto-coloring mode if colors not provided:
            - 'height': Color by z coordinate
            - 'intensity': Color by reflectance
            - 'depth': Color by distance from origin
            - 'uniform': Single color
        window_name: Visualization window title.
        point_size: Point rendering size.
        background_color: Background color (RGB, 0-1).
        show_axes: Whether to show coordinate axes.
        boxes: Optional list of (8, 3) box corners to display.
        box_colors: Optional colors for boxes.

    Raises:
        ImportError: If Open3D is not installed.
    """
    if not HAS_OPEN3D:
        raise ImportError(
            "Open3D is required for 3D visualization. "
            "Install with: pip install open3d"
        )

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Set colors
    if colors is not None:
        # Ensure colors are in [0, 1] range
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Auto-color based on mode
        if color_mode == "height":
            colors_uint8 = height_to_color(points[:, 2])
        elif color_mode == "intensity":
            colors_uint8 = intensity_to_color(points[:, 3])
        elif color_mode == "depth":
            depth = np.linalg.norm(points[:, :3], axis=1)
            colors_uint8 = depth_to_color(depth, 0, 70)
        else:  # uniform
            colors_uint8 = np.full((len(points), 3), [0, 255, 127], dtype=np.uint8)

        pcd.colors = o3d.utility.Vector3dVector(colors_uint8 / 255.0)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # Add point cloud
    vis.add_geometry(pcd)

    # Add coordinate axes
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        vis.add_geometry(axes)

    # Add 3D boxes if provided
    if boxes is not None:
        for i, corners in enumerate(boxes):
            # Create line set for box
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
                [4, 5], [5, 6], [6, 7], [7, 4],  # top
                [0, 4], [1, 5], [2, 6], [3, 7],  # pillars
            ]

            if box_colors and i < len(box_colors):
                color = box_colors[i]
                if max(color) > 1:
                    color = tuple(c / 255 for c in color)
            else:
                color = (0, 1, 0)

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

            vis.add_geometry(line_set)

    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array(background_color)

    # Set viewpoint (look at scene from above-behind)
    ctr = vis.get_view_control()
    ctr.set_lookat([20, 0, 0])  # Look at point 20m ahead
    ctr.set_front([-0.5, 0, 0.5])  # Camera direction
    ctr.set_up([0, 0, 1])  # Up direction
    ctr.set_zoom(0.3)

    # Run visualizer
    vis.run()
    vis.destroy_window()


class PointCloudVisualizer:
    """
    Interactive point cloud visualizer using Open3D.

    Supports updating point cloud in real-time for streaming data.

    Usage:
        viz = PointCloudVisualizer()
        viz.show(points)  # Blocking

        # Or for real-time updates:
        viz.start()
        for points in stream:
            viz.update(points)
        viz.stop()
    """

    def __init__(
        self,
        window_name: str = "Point Cloud Viewer",
        point_size: float = 2.0,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    ):
        """
        Initialize visualizer.

        Args:
            window_name: Window title.
            point_size: Point rendering size.
            background_color: Background RGB color (0-1).
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required. Install with: pip install open3d")

        self.window_name = window_name
        self.point_size = point_size
        self.background_color = background_color

        self._vis = None
        self._pcd = None
        self._initialized = False

    def show(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        color_mode: str = "height",
    ) -> None:
        """
        Show point cloud (blocking).

        Args:
            points: (N, 4) point cloud.
            colors: Optional (N, 3) RGB colors [0-1].
            color_mode: Auto-coloring mode.
        """
        visualize_pointcloud(
            points,
            colors=colors,
            color_mode=color_mode,
            window_name=self.window_name,
            point_size=self.point_size,
            background_color=self.background_color,
        )

    def start(self) -> None:
        """Start the visualizer for real-time updates."""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name=self.window_name, width=1280, height=720)

        # Initialize with empty point cloud
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        # Set options
        opt = self._vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array(self.background_color)

        self._initialized = True

    def update(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        color_mode: str = "height",
    ) -> bool:
        """
        Update displayed point cloud.

        Args:
            points: (N, 4) point cloud.
            colors: Optional (N, 3) RGB colors [0-1].
            color_mode: Auto-coloring mode.

        Returns:
            False if window was closed.
        """
        if not self._initialized:
            self.start()

        # Update points
        self._pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Update colors
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            self._pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            if color_mode == "height":
                c = height_to_color(points[:, 2]) / 255.0
            elif color_mode == "intensity":
                c = intensity_to_color(points[:, 3]) / 255.0
            else:
                c = np.full((len(points), 3), [0, 1, 0.5])
            self._pcd.colors = o3d.utility.Vector3dVector(c)

        # Update visualizer
        self._vis.update_geometry(self._pcd)
        return self._vis.poll_events()

    def stop(self) -> None:
        """Stop the visualizer."""
        if self._vis:
            self._vis.destroy_window()
            self._vis = None
            self._pcd = None
            self._initialized = False
