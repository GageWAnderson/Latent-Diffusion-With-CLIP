import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


class ImageFolderDataset(Dataset):
    def __init__(self,
                 image_dir,
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5):
        self.image_dir = image_dir
        self.size = size
        self.interpolation = {
            "linear": PIL.Image.BILINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(os.path.join(root, fname))
        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # center crop to square
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        return {"image": (image / 127.5 - 1.0).astype(np.float32)}


class ImageFolderTrain(ImageFolderDataset):
    def __init__(self, **kwargs):
        super().__init__(flip_p=0.5, **kwargs)


class ImageFolderValidation(ImageFolderDataset):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)


class ImageCaptionDataset(Dataset):
    """Image + text caption pairs. Expects image files with matching .txt sidecar files."""
    def __init__(self,
                 image_dir,
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5):
        self.size = size
        self.interpolation = {
            "linear": PIL.Image.BILINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.image_paths = []
        self.caption_paths = []
        for root, _, files in os.walk(image_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    img_path = os.path.join(root, fname)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.isfile(txt_path):
                        self.image_paths.append(img_path)
                        self.caption_paths.append(txt_path)
        self.image_paths.sort()
        self.caption_paths.sort()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image+caption pairs found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        with open(self.caption_paths[i], "r") as f:
            caption = f.read().strip()

        return {
            "image": (image / 127.5 - 1.0).astype(np.float32),
            "caption": caption,
        }


class ImageCaptionTrain(ImageCaptionDataset):
    def __init__(self, **kwargs):
        super().__init__(flip_p=0.5, **kwargs)


class ImageCaptionValidation(ImageCaptionDataset):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
