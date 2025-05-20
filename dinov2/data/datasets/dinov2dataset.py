import logging
from pathlib import Path
from typing import Callable, Optional, Union

from dinov2.data.datasets.extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']


class DINOv2Dataset(ExtendedVisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        # Load image paths and corresponding targets (class labels)
        self.image_paths = []
        self.targets = []
        
        # Collect all images and their labels (folder names as class labels)
        for label, class_folder in enumerate(Path(root).iterdir()):
            if class_folder.is_dir():  # Ensure it's a directory
                for image_file in class_folder.glob('*'):
                    if image_file.suffix.lower() in IMG_EXTS:
                        self.image_paths.append(image_file)
                        self.targets.append(label)

    def get_image_data(self, index: int) -> bytes:
        """Read image file and return it as bytes."""
        img_path = self.image_paths[index]
        with open(img_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Optional[Target]:
        """Return the target label for the image at the given index."""
        return self.targets[index]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)
