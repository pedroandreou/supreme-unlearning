"""
Datasets used for the experiments (CIFAR and Celebrity Faces)
"""

import os
import random
from typing import Any, Tuple
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder, Caltech101 as TorchCaltech101
from torchvision.datasets.utils import download_and_extract_archive
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from dotenv import find_dotenv, load_dotenv
# import kaggle

# load_dotenv(find_dotenv())
# os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
# os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
# os.environ["KAGGLE_CONFIG_DIR"] = os.getenv("KAGGLE_CONFIG_DIR")

# ============================================================================
# Per-dataset normalization constants (computed via supreme/utils/compute_dataset_stats.py)
# ResNet18 (trained from scratch): uses dataset-specific stats
# ViT (pretrained on ImageNet): uses ImageNet stats regardless of dataset
# ============================================================================
CIFAR100_MEAN = (0.5070753693580627, 0.4865487813949585, 0.44091784954071045)
CIFAR100_STD = (0.2673334777355194, 0.25643861293792725, 0.2761504352092743)

# CIFAR-20 is CIFAR-100 with superclass labels (same images, same stats)
CIFAR20_MEAN = CIFAR100_MEAN
CIFAR20_STD = CIFAR100_STD

CIFAR10_MEAN = (0.4913999140262604, 0.48215872049331665, 0.4465313255786896)
CIFAR10_STD = (0.24703191220760345, 0.243484228849411, 0.2615869343280792)

PINS_MEAN = (0.516226053237915, 0.4191001355648041, 0.37332481145858765)
PINS_STD = (0.2862764596939087, 0.2552863657474518, 0.2460651993751526)

CALTECH_MEAN = (0.5458726286888123, 0.5287854075431824, 0.5021448731422424)
CALTECH_STD = (0.2954096496105194, 0.2890913784503937, 0.3031054437160492)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Custom transform to convert grayscale to RGB (picklable, unlike lambda)
class GrayscaleToRGB:
    """Convert grayscale images to RGB by repeating channels."""
    def __call__(self, x):
        if x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ============================================================================
# Transform builders: return fresh transform lists with the given normalization
# ============================================================================
def _build_train_transforms(mean, std):
    """Training transforms with augmentation (ResNet, from scratch)."""
    return [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

def _build_unlearning_transforms(mean, std):
    """Unlearning transforms (no augmentation)."""
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

def _build_test_transforms(mean, std):
    """Test/eval transforms (no augmentation)."""
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

# ViT-specific transforms: resize first to 256 then crop to 224 (standard ViT fine-tuning recipe)
# Always uses ImageNet normalization to match the pretrained google/vit-base-patch16-224 backbone
transform_train_vit = [
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]

transform_test_vit = [
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]


class PinsFaceRecognition(ImageFolder):
    def __init__(self, root, train, unlearning, download, img_size=32, model_name=None):
        self.root = root

        is_vit = (model_name == "ViT")
        if is_vit:
            if train and not unlearning:
                transform = list(transform_train_vit)
            else:
                transform = list(transform_test_vit)
        else:
            if train:
                if unlearning:
                    transform = _build_unlearning_transforms(PINS_MEAN, PINS_STD)
                else:
                    transform = _build_train_transforms(PINS_MEAN, PINS_STD)
            else:
                transform = _build_test_transforms(PINS_MEAN, PINS_STD)
            transform.insert(0, transforms.Resize((36, 36), antialias=True))
            transform.append(transforms.Resize((img_size, img_size), antialias=True))
        transform = transforms.Compose(transform)

        super().__init__(self.root, transform)

    # def download(self):
    #     print(f"Entering download method. Root: {self.root}")
    #     if not os.path.exists(self.root) or not os.listdir(self.root):
    #         print("Downloading PinsFaceRecognition dataset...")
    #         try:
    #             dataset_name = "hereisburak/pins-face-recognition"
    #             kaggle.api.dataset_download_files(
    #                 dataset_name, path=self.root, unzip=True
    #             )
    #             print("Contents after download:", os.listdir(self.root))
    #             print("Download and extraction complete.")
    #         except Exception as e:
    #             print(f"An error occurred during download: {e}")
    #     else:
    #         print(
    #             f"PinsFaceRecognition dataset already downloaded. Contents: {os.listdir(self.root)}"
    #         )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar100(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, model_name=None):
        is_vit = (model_name == "ViT")
        if is_vit:
            if train and not unlearning:
                transform = list(transform_train_vit)
            else:
                transform = list(transform_test_vit)
        else:
            if train:
                if unlearning:
                    transform = _build_unlearning_transforms(CIFAR100_MEAN, CIFAR100_STD)
                else:
                    transform = _build_train_transforms(CIFAR100_MEAN, CIFAR100_STD)
            else:
                transform = _build_test_transforms(CIFAR100_MEAN, CIFAR100_STD)
            transform.append(transforms.Resize(img_size, antialias=True))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar20(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32, model_name=None):
        is_vit = (model_name == "ViT")
        if is_vit:
            if train and not unlearning:
                transform = list(transform_train_vit)
            else:
                transform = list(transform_test_vit)
        else:
            if train:
                if unlearning:
                    transform = _build_unlearning_transforms(CIFAR20_MEAN, CIFAR20_STD)
                else:
                    transform = _build_train_transforms(CIFAR20_MEAN, CIFAR20_STD)
            else:
                transform = _build_test_transforms(CIFAR20_MEAN, CIFAR20_STD)
            transform.append(transforms.Resize(img_size, antialias=True))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

        # Create reverse mapping from fine label to coarse label for fast lookups
        self.fine_to_coarse = {}
        for coarse_label, fine_labels in self.coarse_map.items():
            for fine_label in fine_labels:
                self.fine_to_coarse[fine_label] = coarse_label

        # Create coarse_targets attribute for fast classwise indexing (line 55 in unlearning_utils.py)
        # This ensures the fast path in get_classwise_indices uses coarse labels, not fine labels
        self.coarse_targets = [self.fine_to_coarse[fine_label] for fine_label in self.targets]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y is not None:
                break
        if coarse_y is None:
            print(y)
            assert coarse_y is not None
        return x, y, coarse_y


class Cifar10(CIFAR10):
    def __init__(self, root, train, unlearning, download, img_size=32, model_name=None):
        is_vit = (model_name == "ViT")
        if is_vit:
            if train and not unlearning:
                transform = list(transform_train_vit)
            else:
                transform = list(transform_test_vit)
        else:
            if train:
                if unlearning:
                    transform = _build_unlearning_transforms(CIFAR10_MEAN, CIFAR10_STD)
                else:
                    transform = _build_train_transforms(CIFAR10_MEAN, CIFAR10_STD)
            else:
                transform = _build_test_transforms(CIFAR10_MEAN, CIFAR10_STD)
            transform.append(transforms.Resize(img_size, antialias=True))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Caltech101(TorchCaltech101):
    def __init__(self, root, train, unlearning, download, img_size=32, train_split=0.8, model_name=None):
        is_vit = (model_name == "ViT")
        if is_vit:
            if train and not unlearning:
                transform = list(transform_train_vit)
            else:
                transform = list(transform_test_vit)
            # Caltech101 has some grayscale images - insert GrayscaleToRGB after ToTensor but before Normalize
            normalize_idx = next(i for i, t in enumerate(transform) if isinstance(t, transforms.Normalize))
            transform.insert(normalize_idx, GrayscaleToRGB())
        else:
            if train:
                if unlearning:
                    transform = _build_unlearning_transforms(CALTECH_MEAN, CALTECH_STD)
                else:
                    transform = _build_train_transforms(CALTECH_MEAN, CALTECH_STD)
            else:
                transform = _build_test_transforms(CALTECH_MEAN, CALTECH_STD)
            # Caltech101 has variable-size images, so resize to fixed size first
            transform.insert(0, transforms.Resize((36, 36), antialias=True))
            normalize_idx = next(i for i, t in enumerate(transform) if isinstance(t, transforms.Normalize))
            transform.insert(normalize_idx, GrayscaleToRGB())
            transform.append(transforms.Resize((img_size, img_size), antialias=True))
        transform = transforms.Compose(transform)

        # Initialize the parent dataset
        # Pass download parameter - our overridden download() method will be called if True
        super().__init__(root=root, download=download, transform=transform)

        # Implement train/test split
        self.train = train
        self.train_split = train_split

        # Note: Random seed is already set globally via fabric.seed_everything() in set_seeds()
        # at the start of train_main.py/unlearn_main.py, so no local seed setting needed here.

        # Create per-class train/test split
        self._create_train_test_split()

    def download(self):
        """Override download method with working download source from Caltech"""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        print("="*80)
        print("Downloading Caltech101 dataset...")
        print("="*80)

        os.makedirs(self.root, exist_ok=True)

        # Use direct download from Caltech repository
        # Source: https://data.caltech.edu/records/mzrjq-6wc02
        url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
        filename = "caltech-101.zip"
        md5_hash = "3138e1922a9193bfa496528edbbc45d0"

        print(f"Downloading from official Caltech repository...")
        print(f"URL: {url}")
        print(f"File size: ~137 MB")
        print()

        try:
            download_and_extract_archive(
                url,
                self.root,
                filename=filename,
                md5=md5_hash,
                remove_finished=True
            )
            print("✓ Main dataset downloaded and extracted successfully!")

            # The zip file contains nested compressed files that need extraction:
            # caltech-101/101_ObjectCategories.tar.gz -> needs extraction
            # caltech-101/Annotations.tar -> needs extraction
            extracted_dir = os.path.join(self.root, "caltech-101")
            if os.path.exists(extracted_dir):
                print("Extracting nested archives...")
                import shutil
                import tarfile

                # Extract 101_ObjectCategories.tar.gz
                categories_tar = os.path.join(extracted_dir, "101_ObjectCategories.tar.gz")
                if os.path.exists(categories_tar):
                    print("Extracting 101_ObjectCategories.tar.gz...")
                    with tarfile.open(categories_tar, 'r:gz') as tar:
                        tar.extractall(self.root)
                    print("✓ Extracted 101_ObjectCategories")
                    os.remove(categories_tar)

                # Extract Annotations.tar
                annotations_tar = os.path.join(extracted_dir, "Annotations.tar")
                if os.path.exists(annotations_tar):
                    print("Extracting Annotations.tar...")
                    with tarfile.open(annotations_tar, 'r') as tar:
                        tar.extractall(self.root)
                    print("✓ Extracted Annotations")
                    os.remove(annotations_tar)

                # Clean up the extracted directory and __MACOSX
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)
                macosx_dir = os.path.join(self.root, "__MACOSX")
                if os.path.exists(macosx_dir):
                    shutil.rmtree(macosx_dir)

            # Verify the download
            if self._check_integrity():
                print("✓ Dataset verified successfully!")
                print("="*80)
                return
            else:
                raise RuntimeError("Download completed but verification failed")

        except Exception as e:
            print(f"✗ Download failed: {e}")
            print()
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"Automatic download failed.\n"
                f"{'='*80}\n\n"
                f"MANUAL DOWNLOAD INSTRUCTIONS:\n"
                f"1. Visit: https://data.caltech.edu/records/mzrjq-6wc02\n\n"
                f"2. Download the file: caltech-101.zip (137.4 MB)\n"
                f"   Direct link: {url}\n\n"
                f"3. Extract the zip file to: {self.root}\n\n"
                f"4. Ensure the directory structure is:\n"
                f"   {self.root}/caltech-101/101_ObjectCategories/\n"
                f"   {self.root}/caltech-101/101_ObjectCategories/accordion/\n"
                f"   {self.root}/caltech-101/101_ObjectCategories/airplanes/\n"
                f"   ... etc.\n\n"
                f"5. Then rename 'caltech-101' to match expected structure if needed\n\n"
                f"Original error: {e}\n"
                f"{'='*80}\n"
            )

    def _create_train_test_split(self):
        """Create deterministic train/test split maintaining class distribution.

        Unlike CIFAR which has predefined train/test sets from torchvision,
        Caltech101 loads all images, so we create a deterministic split here.
        No shuffling to ensure train and test sets are always disjoint.
        """
        # Group indices by class
        class_indices = {}
        for idx, class_idx in enumerate(self.y):
            if class_idx not in class_indices:
                class_indices[class_idx] = []
            class_indices[class_idx].append(idx)

        # Split each class deterministically (first train_split% -> train, rest -> test)
        train_indices = []
        test_indices = []
        for class_idx in sorted(class_indices.keys()):
            indices = sorted(class_indices[class_idx])
            split_point = int(len(indices) * self.train_split)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])

        # Store the appropriate indices based on train/test mode
        if self.train:
            self.data_indices = train_indices
        else:
            self.data_indices = test_indices

        print(f"Caltech101 {'Train' if self.train else 'Test'} split: {len(self.data_indices)} samples")

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        # Map the index to the actual dataset index
        actual_index = self.data_indices[index]
        x, y = super().__getitem__(actual_index)
        return x, torch.Tensor([]), y


class CombinedForgetRetainDataset(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y
