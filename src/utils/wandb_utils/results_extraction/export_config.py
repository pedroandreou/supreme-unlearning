#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Central configuration file for the W&B metrics export pipeline.

This file is organized into sections:
1. SHARED CONFIGURATION - Constants used across the export pipeline
2. WANDB EXPORTER CONFIGURATION - Settings consumed by orchestrate_wandb_export.sh
"""

# ============================================================================
# SECTION 1: SHARED CONFIGURATION
# ============================================================================

# Experiment prefix configuration
EXPERIMENT_PREFIXES = ["R6_UNLEARNING", "R7_UNLEARNING", "P1_UNLEARNING"]
DEFAULT_EXPERIMENT_PREFIX = "P1_UNLEARNING"  # Default to P1 for current experiments

# Seeds used across all experiments (R7 uses seeds 60-69)
SEEDS = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]

# Method aliases for canonical name matching (used for name normalization in both scripts)
METHOD_ALIASES = {
    "original": {"original"},
    "retrain": {"retrain"},
    "ft": {"finetune"},
    "badt": {"bad_teacher"},
    "unsir": {"unsir"},
    "rl": {"random_labeling"},
    "ssd": {"ssd"},
    "lfssd": {"lfssd"},
}

# Derived path configuration based on experiment prefix
# Extracts short prefix (e.g., "R7" from "R7_UNLEARNING") for directory naming
def _get_prefix_short(prefix: str = DEFAULT_EXPERIMENT_PREFIX) -> str:
    """Extract short prefix (R6/R7) from full experiment prefix."""
    return prefix.split("_")[0]

# Path configuration for data directories (relative to results_extraction/)
COMBINED_RESULTS_DIR = f"combined_results_{_get_prefix_short()}"
WANDB_METRICS_SUMMARY_DIR = f"wandb_metrics_summary_{_get_prefix_short()}"

# ============================================================================
# SECTION 2: WANDB METRICS EXPORTER CONFIGURATION
# ============================================================================

# All models
MODELS = ["ResNet18", "ViT"]

# All datasets
# Order: CIFAR20 first, then CIFAR100, then CIFAR10, then additional datasets
DATASETS = ["Cifar20", "Cifar100", "Cifar10", "Caltech101", "PinsFaceRecognition"]

# Custom table order for LaTeX output (dataset, strategy pairs)
# Groups: 1) Fullclass, 2) Subclass, 3) Random
# Within each group, tables are ordered to maintain logical flow
TABLE_ORDER = [
    # --- FULLCLASS TABLES ---
    ("Cifar20", "fullclass"),
    ("Cifar100", "fullclass"),
    ("Caltech101", "fullclass"),
    ("PinsFaceRecognition", "fullclass"),
    # --- SUBCLASS TABLES ---
    ("Cifar20", "subclass"),
    # --- RANDOM TABLES ---
    ("Cifar10", "random_"),
    ("PinsFaceRecognition", "random_"),
    ("Caltech101", "random_"),
]

# Section breaks for LaTeX output (insert \clearpage\newpage after these)
TABLE_SECTION_BREAKS = {
    ("PinsFaceRecognition", "fullclass"): "End of Fullclass Section",
    ("Cifar20", "subclass"): "End of Subclass Section",
}

# Dataset-strategy-class mappings (for all-possible mode in export script)
DATASET_CLASSES = {
    "Cifar20_fullclass": ["electrical_devices", "natural_scenes", "people", "vehicle2", "veg"],
    "Cifar20_subclass": ["baby", "lamp", "mushroom", "rocket", "sea"],  # CIFAR100 classes
    "Cifar100_fullclass": ["baby", "lamp", "mushroom", "rocket", "sea"],
    "PinsFaceRecognition_fullclass": ["1", "10", "20", "30", "40"],
    "PinsFaceRecognition_random_": ["0.001", "0.005", "0.01", "0.05", "0.1"],  # 5 forget percentages
    "Cifar10_random_": ["0.001", "0.005", "0.01", "0.05", "0.1"],  # 5 forget percentages
    "Caltech101_fullclass": ["airplanes", "car_side", "chair", "elephant", "lamp"],
    "Caltech101_random_": ["0.001", "0.005", "0.01", "0.05", "0.1"],  # 5 forget percentages
}

# Dataset-strategy mappings
DATASET_STRATEGIES = {
    "Cifar20": ["fullclass", "subclass"],
    "Cifar100": ["fullclass"],
    "PinsFaceRecognition": ["fullclass", "random_"],
    "Cifar10": ["random_"],
    "Caltech101": ["fullclass", "random_"],
}

# Active metrics for W&B export
# These metrics determine which Excel files are created by wandb_metrics_exporter.py
ACTIVE_METRICS = [
    "accuracy",
    "loss",
    "zrf",
    "membership_inference_attack",
    "activation_distance",
    "completeness",
    "jsdiv",
    "layerwise_distance",
    "time",
]

# -----------------------------------------------------------------------------

# Default precision setting
DEFAULT_PRECISION = "32-true"

# Experiment directory prefix for wandb_metrics_summary
EXPERIMENT_DIR_PREFIX = "P1_UNLEARNING_"

# Existing project combinations (for orchestrate_wandb_export.sh)
# Format: (model, dataset, strategy, class_or_perc, precision)
EXISTING_PROJECTS = [
    # --- CIFAR10 Random (5 forget percentages) ---
    ("ResNet18", "Cifar10", "random_", "0.001", "32-true"),
    ("ResNet18", "Cifar10", "random_", "0.005", "32-true"),
    ("ResNet18", "Cifar10", "random_", "0.01", "32-true"),
    ("ResNet18", "Cifar10", "random_", "0.05", "32-true"),
    ("ResNet18", "Cifar10", "random_", "0.1", "32-true"),
    ("ViT", "Cifar10", "random_", "0.001", "32-true"),
    ("ViT", "Cifar10", "random_", "0.005", "32-true"),
    ("ViT", "Cifar10", "random_", "0.01", "32-true"),
    ("ViT", "Cifar10", "random_", "0.05", "32-true"),
    ("ViT", "Cifar10", "random_", "0.1", "32-true"),
    # --- CIFAR20 Fullclass (5 classes) ---
    ("ResNet18", "Cifar20", "fullclass", "electrical_devices", "32-true"),
    ("ResNet18", "Cifar20", "fullclass", "natural_scenes", "32-true"),
    ("ResNet18", "Cifar20", "fullclass", "people", "32-true"),
    ("ResNet18", "Cifar20", "fullclass", "vehicle2", "32-true"),
    ("ResNet18", "Cifar20", "fullclass", "veg", "32-true"),
    ("ViT", "Cifar20", "fullclass", "electrical_devices", "32-true"),
    ("ViT", "Cifar20", "fullclass", "natural_scenes", "32-true"),
    ("ViT", "Cifar20", "fullclass", "people", "32-true"),
    ("ViT", "Cifar20", "fullclass", "vehicle2", "32-true"),
    ("ViT", "Cifar20", "fullclass", "veg", "32-true"),
    # --- CIFAR20 Subclass ---
    ("ResNet18", "Cifar20", "subclass", "baby", "32-true"),
    ("ResNet18", "Cifar20", "subclass", "lamp", "32-true"),
    ("ResNet18", "Cifar20", "subclass", "mushroom", "32-true"),
    ("ResNet18", "Cifar20", "subclass", "rocket", "32-true"),
    ("ResNet18", "Cifar20", "subclass", "sea", "32-true"),
    ("ViT", "Cifar20", "subclass", "baby", "32-true"),
    ("ViT", "Cifar20", "subclass", "lamp", "32-true"),
    ("ViT", "Cifar20", "subclass", "mushroom", "32-true"),
    ("ViT", "Cifar20", "subclass", "rocket", "32-true"),
    ("ViT", "Cifar20", "subclass", "sea", "32-true"),
    # --- CIFAR100 Fullclass ---
    ("ResNet18", "Cifar100", "fullclass", "baby", "32-true"),
    ("ResNet18", "Cifar100", "fullclass", "lamp", "32-true"),
    ("ResNet18", "Cifar100", "fullclass", "mushroom", "32-true"),
    ("ResNet18", "Cifar100", "fullclass", "rocket", "32-true"),
    ("ResNet18", "Cifar100", "fullclass", "sea", "32-true"),
    ("ViT", "Cifar100", "fullclass", "baby", "32-true"),
    ("ViT", "Cifar100", "fullclass", "lamp", "32-true"),
    ("ViT", "Cifar100", "fullclass", "mushroom", "32-true"),
    ("ViT", "Cifar100", "fullclass", "rocket", "32-true"),
    ("ViT", "Cifar100", "fullclass", "sea", "32-true"),
    # --- PinsFaceRecognition Fullclass ---
    ("ResNet18", "PinsFaceRecognition", "fullclass", "1", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "fullclass", "10", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "fullclass", "20", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "fullclass", "30", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "fullclass", "40", "32-true"),
    ("ViT", "PinsFaceRecognition", "fullclass", "1", "32-true"),
    ("ViT", "PinsFaceRecognition", "fullclass", "10", "32-true"),
    ("ViT", "PinsFaceRecognition", "fullclass", "20", "32-true"),
    ("ViT", "PinsFaceRecognition", "fullclass", "30", "32-true"),
    ("ViT", "PinsFaceRecognition", "fullclass", "40", "32-true"),
    # --- PinsFaceRecognition Random (5 forget percentages) ---
    ("ResNet18", "PinsFaceRecognition", "random_", "0.001", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "random_", "0.005", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "random_", "0.01", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "random_", "0.05", "32-true"),
    ("ResNet18", "PinsFaceRecognition", "random_", "0.1", "32-true"),
    ("ViT", "PinsFaceRecognition", "random_", "0.001", "32-true"),
    ("ViT", "PinsFaceRecognition", "random_", "0.005", "32-true"),
    ("ViT", "PinsFaceRecognition", "random_", "0.01", "32-true"),
    ("ViT", "PinsFaceRecognition", "random_", "0.05", "32-true"),
    ("ViT", "PinsFaceRecognition", "random_", "0.1", "32-true"),
    # --- Caltech101 Fullclass ---
    ("ResNet18", "Caltech101", "fullclass", "airplanes", "32-true"),
    ("ResNet18", "Caltech101", "fullclass", "car_side", "32-true"),
    ("ResNet18", "Caltech101", "fullclass", "chair", "32-true"),
    ("ResNet18", "Caltech101", "fullclass", "elephant", "32-true"),
    ("ResNet18", "Caltech101", "fullclass", "lamp", "32-true"),
    ("ViT", "Caltech101", "fullclass", "airplanes", "32-true"),
    ("ViT", "Caltech101", "fullclass", "car_side", "32-true"),
    ("ViT", "Caltech101", "fullclass", "chair", "32-true"),
    ("ViT", "Caltech101", "fullclass", "elephant", "32-true"),
    ("ViT", "Caltech101", "fullclass", "lamp", "32-true"),
    # --- Caltech101 Random (5 forget percentages) ---
    ("ResNet18", "Caltech101", "random_", "0.001", "32-true"),
    ("ResNet18", "Caltech101", "random_", "0.005", "32-true"),
    ("ResNet18", "Caltech101", "random_", "0.01", "32-true"),
    ("ResNet18", "Caltech101", "random_", "0.05", "32-true"),
    ("ResNet18", "Caltech101", "random_", "0.1", "32-true"),
    ("ViT", "Caltech101", "random_", "0.001", "32-true"),
    ("ViT", "Caltech101", "random_", "0.005", "32-true"),
    ("ViT", "Caltech101", "random_", "0.01", "32-true"),
    ("ViT", "Caltech101", "random_", "0.05", "32-true"),
    ("ViT", "Caltech101", "random_", "0.1", "32-true"),
]

