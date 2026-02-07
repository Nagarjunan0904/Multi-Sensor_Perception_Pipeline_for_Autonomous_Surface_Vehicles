"""Camera data loader for RGB images."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class CameraLoader:
    """Load and preprocess camera images from KITTI dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "training",
        image_dir: str = "image_2",
        normalize: bool = False,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize the camera loader.

        Args:
            data_root: Root directory of the KITTI dataset.
            split: Dataset split ('training' or 'testing').
            image_dir: Subdirectory containing images.
            normalize: Whether to normalize images with ImageNet stats.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_dir = image_dir
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.image_path = self.data_root / split / image_dir
        self._validate_path()
        self._index_images()

    def _validate_path(self) -> None:
        """Validate that the image directory exists."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")

    def _index_images(self) -> None:
        """Index all available images."""
        self.image_files = sorted(self.image_path.glob("*.png"))
        if not self.image_files:
            self.image_files = sorted(self.image_path.glob("*.jpg"))

    def __len__(self) -> int:
        """Return the number of available images."""
        return len(self.image_files)

    def __getitem__(self, index: int) -> np.ndarray:
        """Load image by index."""
        return self.load_image(index)

    def load_image(
        self,
        index: Union[int, str],
        color_format: str = "BGR",
    ) -> np.ndarray:
        """
        Load a single image.

        Args:
            index: Image index (int) or filename (str).
            color_format: Output color format ('BGR' or 'RGB').

        Returns:
            Image as numpy array (H, W, C).
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self.image_files):
                raise IndexError(f"Image index {index} out of range [0, {len(self) - 1}]")
            image_path = self.image_files[index]
        else:
            image_path = self.image_path / index
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")

        if color_format.upper() == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def preprocess(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image (H, W, C) in BGR format.
            target_size: Optional (width, height) to resize to.

        Returns:
            Preprocessed image as float32 array.
        """
        if target_size is not None:
            image = cv2.resize(image, target_size)

        image = image.astype(np.float32) / 255.0

        if self.normalize:
            image = (image - self.mean) / self.std

        return image

    def get_image_size(self, index: int = 0) -> Tuple[int, int]:
        """
        Get image dimensions.

        Args:
            index: Image index to check.

        Returns:
            Tuple of (width, height).
        """
        image = self.load_image(index)
        return image.shape[1], image.shape[0]

    def get_frame_id(self, index: int) -> str:
        """
        Get frame ID for given index.

        Args:
            index: Image index.

        Returns:
            Frame ID string (e.g., '000000').
        """
        return self.image_files[index].stem
