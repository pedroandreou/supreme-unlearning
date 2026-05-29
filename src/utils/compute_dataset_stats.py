"""
Compute per-channel mean and std for all datasets.
Used when adding a new dataset to determine the correct normalization constants.
Run from project root: python src/utils/compute_dataset_stats.py
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Handles grayscale images in Caltech101
class GrayscaleToRGB:
    def __call__(self, x):
        if x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x


def compute_mean_std(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0

    for batch in loader:
        # Handle different return formats: (img,), (img, label), (img, _, label), etc.
        images = batch[0]
        b, c, h, w = images.shape
        n_pixels += b * h * w
        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])

    mean /= n_pixels
    std = (std / n_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- PinsFaceRecognition ---
    pins_root = os.path.join(script_dir, "../datasets/data/105_classes_pins_dataset")
    if os.path.exists(pins_root):
        print("=" * 60)
        print("PinsFaceRecognition")
        print("=" * 60)
        ds = datasets.ImageFolder(
            pins_root,
            transform=transforms.Compose([
                transforms.Resize((36, 36), antialias=True),
                transforms.ToTensor(),
            ]),
        )
        print(f"  Samples: {len(ds)}")
        mean, std = compute_mean_std(ds)
        print(f"  Mean: ({mean[0]:.10f}, {mean[1]:.10f}, {mean[2]:.10f})")
        print(f"  Std:  ({std[0]:.10f}, {std[1]:.10f}, {std[2]:.10f})")
        print()
        print(f"  # For datasets.py:")
        print(f"  PINS_MEAN = ({mean[0]}, {mean[1]}, {mean[2]})")
        print(f"  PINS_STD = ({std[0]}, {std[1]}, {std[2]})")
    else:
        print(f"PinsFaceRecognition not found at: {pins_root}")

    print()

    # --- Caltech101 ---
    caltech_root = os.path.join(script_dir, "../datasets/data")
    if os.path.exists(os.path.join(caltech_root, "caltech-101")) or os.path.exists(
        os.path.join(caltech_root, "101_ObjectCategories")
    ) or os.path.exists(os.path.join(caltech_root, "caltech101")):
        print("=" * 60)
        print("Caltech101")
        print("=" * 60)
        ds = datasets.Caltech101(
            root=caltech_root,
            download=False,
            transform=transforms.Compose([
                transforms.Resize((36, 36), antialias=True),
                transforms.ToTensor(),
                GrayscaleToRGB(),
            ]),
        )
        print(f"  Samples: {len(ds)}")
        mean, std = compute_mean_std(ds)
        print(f"  Mean: ({mean[0]:.10f}, {mean[1]:.10f}, {mean[2]:.10f})")
        print(f"  Std:  ({std[0]:.10f}, {std[1]:.10f}, {std[2]:.10f})")
        print()
        print(f"  # For datasets.py:")
        print(f"  CALTECH_MEAN = ({mean[0]}, {mean[1]}, {mean[2]})")
        print(f"  CALTECH_STD = ({std[0]}, {std[1]}, {std[2]})")
    else:
        print(f"Caltech101 not found at: {caltech_root}")

    print()

    cifar_root = os.path.join(script_dir, "../datasets/data/cifar")

    # --- CIFAR-10 ---
    print("=" * 60)
    print("CIFAR-10")
    print("  Code uses CIFAR-100 stats: mean (0.5071, 0.4865, 0.4409) / std (0.2673, 0.2564, 0.2762)")
    print("=" * 60)
    try:
        ds = datasets.CIFAR10(
            root=cifar_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        print(f"  Samples: {len(ds)}")
        mean, std = compute_mean_std(ds)
        print(f"  Mean: ({mean[0]:.10f}, {mean[1]:.10f}, {mean[2]:.10f})")
        print(f"  Std:  ({std[0]:.10f}, {std[1]:.10f}, {std[2]:.10f})")
        print()
        print(f"  # For datasets.py:")
        print(f"  CIFAR10_MEAN = ({mean[0]}, {mean[1]}, {mean[2]})")
        print(f"  CIFAR10_STD = ({std[0]}, {std[1]}, {std[2]})")
    except Exception as e:
        print(f"  Could not load CIFAR-10: {e}")

    print()

    # --- CIFAR-100 ---
    print("=" * 60)
    print("CIFAR-100")
    print("  Code uses: mean (0.5071, 0.4865, 0.4409) / std (0.2673, 0.2564, 0.2762)")
    print("=" * 60)
    try:
        ds = datasets.CIFAR100(
            root=cifar_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        print(f"  Samples: {len(ds)}")
        mean, std = compute_mean_std(ds)
        print(f"  Mean: ({mean[0]:.10f}, {mean[1]:.10f}, {mean[2]:.10f})")
        print(f"  Std:  ({std[0]:.10f}, {std[1]:.10f}, {std[2]:.10f})")
        print()
        print(f"  # For datasets.py:")
        print(f"  CIFAR100_MEAN = ({mean[0]}, {mean[1]}, {mean[2]})")
        print(f"  CIFAR100_STD = ({std[0]}, {std[1]}, {std[2]})")
    except Exception as e:
        print(f"  Could not load CIFAR-100: {e}")

    print()

    # --- CIFAR-20 (same images as CIFAR-100, just superclass labels) ---
    print("=" * 60)
    print("CIFAR-20 (= CIFAR-100 with superclass labels, should be identical stats)")
    print("=" * 60)
    print("  Same dataset as CIFAR-100, so mean/std are identical by definition.")


if __name__ == "__main__":
    main()
