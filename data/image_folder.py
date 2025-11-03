"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from pathlib import Path
from PIL import Image
import os

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

IMG_EXTENSIONS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.ppm','.webp')

def make_dataset_fast(root_dir, max_images=float('inf'), sort_per_dir=True):
    """Fast recursive listing using os.scandir. Early-stops at max_images."""
    max_images = int(max_images) if max_images != float('inf') else max_images
    out, stack = [], [root_dir]
    exts = tuple(e.lower() for e in IMG_EXTENSIONS)

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except FileNotFoundError:
            continue
        if sort_per_dir:
            entries.sort(key=lambda e: e.name)

        for e in entries:
            if e.is_dir(follow_symlinks=False):
                stack.append(e.path)
            else:
                name = e.name.lower()
                if name.endswith(exts):
                    out.append(e.path)
                    if len(out) >= max_images:
                        return out
    return out

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    dir_path = Path(dir)
    assert dir_path.is_dir(), f"{dir} is not a valid directory"

    for path in dir_path.rglob("*"):
        if path.is_file() and is_image_file(path.name):
            images.append(str(path))
        if len(images) >= max_dataset_size:
            return images
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
