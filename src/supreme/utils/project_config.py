"""
configurations for this project
"""

from datetime import datetime
import os

# ==============================================================================
# ### MODELS ###
# ==============================================================================
# List all models
model_names = [
    "ResNet18",
    "ViT",
    # "your_new_model"  # Add your model name here
]

# ==============================================================================
# ### UNLEARNING METHODS ###
# ==============================================================================
# List of baseline methods
baselines = ["original", "retrain"]

# List of unlearning methods
unlearning_methods = [
    "finetune",
    "bad_teacher",
    "random_labeling",
    "unsir",
    "ssd",
    "lfssd",
    "assd",
    # "neg_grad",
    "scrub",
    "jit",
]

# Combine both lists for use in argument parsing
all_methods = baselines + unlearning_methods

# ==============================================================================
# ### DATASETS ###
# ==============================================================================
dataset_names = [
    "Cifar10",
    "Cifar20",
    "Cifar100",
    "PinsFaceRecognition",
    "Caltech101",
    # "your_new_dataset"  # Add your dataset name here
]

# ==============================================================================
# ### DATASET CLASS MAPPINGS (for unlearning experiments) ###
# ==============================================================================

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar20_dict = {
    "vehicle2": 19,
    "veg": 4,
    "people": 14,
    "electrical_devices": 5,
    "natural_scenes": 10,
}

# Complete CIFAR-20 superclass mappings (all 20 superclasses) - Reference for future use
# cifar20_dict = {
#     "aquatic_mammals": 0,
#     "fish": 1,
#     "flowers": 2,
#     "food_containers": 3,
#     "fruit_and_vegetables": 4,
#     "household_electrical_devices": 5,
#     "household_furniture": 6,
#     "insects": 7,
#     "large_carnivores": 8,
#     "large_man-made_outdoor_things": 9,
#     "large_natural_outdoor_scenes": 10,
#     "large_omnivores_and_herbivores": 11,
#     "medium_mammals": 12,
#     "non-insect_invertebrates": 13,
#     "people": 14,
#     "reptiles": 15,
#     "small_mammals": 16,
#     "trees": 17,
#     "vehicles_1": 18,
#     "vehicles_2": 19,
# }

cifar20_dict_inverted = {v: k for k, v in cifar20_dict.items()}

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar100_dict = {"rocket": 69, "mushroom": 51, "baby": 2, "lamp": 40, "sea": 71}

# Complete CIFAR-100 fine-grained class mappings (all 100 classes) - Reference for future use
# cifar100_dict = {
#     "apple": 0,
#     "aquarium_fish": 1,
#     "baby": 2,
#     "bear": 3,
#     "beaver": 4,
#     "bed": 5,
#     "bee": 6,
#     "beetle": 7,
#     "bicycle": 8,
#     "bottle": 9,
#     "bowl": 10,
#     "boy": 11,
#     "bridge": 12,
#     "bus": 13,
#     "butterfly": 14,
#     "camel": 15,
#     "can": 16,
#     "castle": 17,
#     "caterpillar": 18,
#     "cattle": 19,
#     "chair": 20,
#     "chimpanzee": 21,
#     "clock": 22,
#     "cloud": 23,
#     "cockroach": 24,
#     "couch": 25,
#     "crab": 26,
#     "crocodile": 27,
#     "cup": 28,
#     "dinosaur": 29,
#     "dolphin": 30,
#     "elephant": 31,
#     "flatfish": 32,
#     "forest": 33,
#     "fox": 34,
#     "girl": 35,
#     "hamster": 36,
#     "house": 37,
#     "kangaroo": 38,
#     "keyboard": 39,
#     "lamp": 40,
#     "lawn_mower": 41,
#     "leopard": 42,
#     "lion": 43,
#     "lizard": 44,
#     "lobster": 45,
#     "man": 46,
#     "maple_tree": 47,
#     "motorcycle": 48,
#     "mountain": 49,
#     "mouse": 50,
#     "mushroom": 51,
#     "oak_tree": 52,
#     "orange": 53,
#     "orchid": 54,
#     "otter": 55,
#     "palm_tree": 56,
#     "pear": 57,
#     "pickup_truck": 58,
#     "pine_tree": 59,
#     "plain": 60,
#     "plate": 61,
#     "poppy": 62,
#     "porcupine": 63,
#     "possum": 64,
#     "rabbit": 65,
#     "raccoon": 66,
#     "ray": 67,
#     "road": 68,
#     "rocket": 69,
#     "rose": 70,
#     "sea": 71,
#     "seal": 72,
#     "shark": 73,
#     "shrew": 74,
#     "skunk": 75,
#     "skyscraper": 76,
#     "snail": 77,
#     "snake": 78,
#     "spider": 79,
#     "squirrel": 80,
#     "streetcar": 81,
#     "sunflower": 82,
#     "sweet_pepper": 83,
#     "table": 84,
#     "tank": 85,
#     "telephone": 86,
#     "television": 87,
#     "tiger": 88,
#     "tractor": 89,
#     "train": 90,
#     "trout": 91,
#     "tulip": 92,
#     "turtle": 93,
#     "wardrobe": 94,
#     "whale": 95,
#     "willow_tree": 96,
#     "wolf": 97,
#     "woman": 98,
#     "worm": 99,
# }

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
pins_dict = {"1": 1, "10": 10, "20": 20, "30": 30, "40": 40}

# Complete PinsFaceRecognition class mappings (all 105 person folders) - Reference for future use
# NOTE: ImageFolder sorts alphabetically, so indices match alphabetical order, NOT numeric folder IDs
# pins_dict = {
#     "adriana_lima": 0,
#     "alex_lawther": 1,
#     "alexandra_daddario": 2,
#     "alvaro_morte": 3,
#     "amanda_crew": 4,
#     "andy_samberg": 5,
#     "anne_hathaway": 6,
#     "anthony_mackie": 7,
#     "avril_lavigne": 8,
#     "ben_affleck": 9,
#     "bill_gates": 10,
#     "bobby_morley": 11,
#     "brenton_thwaites": 12,
#     "brian_j._smith": 13,
#     "brie_larson": 14,
#     "chris_evans": 15,
#     "chris_hemsworth": 16,
#     "chris_pratt": 17,
#     "christian_bale": 18,
#     "cristiano_ronaldo": 19,
#     "danielle_panabaker": 20,
#     "dominic_purcell": 21,
#     "dwayne_johnson": 22,
#     "eliza_taylor": 23,
#     "elizabeth_lail": 24,
#     "emilia_clarke": 25,
#     "emma_stone": 26,
#     "emma_watson": 27,
#     "gwyneth_paltrow": 28,
#     "henry_cavil": 29,
#     "hugh_jackman": 30,
#     "inbar_lavi": 31,
#     "irina_shayk": 32,
#     "jake_mcdorman": 33,
#     "jason_momoa": 34,
#     "jennifer_lawrence": 35,
#     "jeremy_renner": 36,
#     "jessica_barden": 37,
#     "jimmy_fallon": 38,
#     "johnny_depp": 39,
#     "josh_radnor": 40,
#     "katharine_mcphee": 41,
#     "katherine_langford": 42,
#     "keanu_reeves": 43,
#     "krysten_ritter": 44,
#     "leonardo_dicaprio": 45,
#     "lili_reinhart": 46,
#     "lindsey_morgan": 47,
#     "lionel_messi": 48,
#     "logan_lerman": 49,
#     "madelaine_petsch": 50,
#     "maisie_williams": 51,
#     "maria_pedraza": 52,
#     "marie_avgeropoulos": 53,
#     "mark_ruffalo": 54,
#     "mark_zuckerberg": 55,
#     "megan_fox": 56,
#     "miley_cyrus": 57,
#     "millie_bobby_brown": 58,
#     "morena_baccarin": 59,
#     "morgan_freeman": 60,
#     "nadia_hilker": 61,
#     "natalie_dormer": 62,
#     "natalie_portman": 63,
#     "neil_patrick_harris": 64,
#     "pedro_alonso": 65,
#     "penn_badgley": 66,
#     "rami_malek": 67,
#     "rebecca_ferguson": 68,
#     "richard_harmon": 69,
#     "rihanna": 70,
#     "robert_de_niro": 71,
#     "robert_downey_jr": 72,
#     "sarah_wayne_callies": 73,
#     "selena_gomez": 74,
#     "shakira_isabel_mebarak": 75,
#     "sophie_turner": 76,
#     "stephen_amell": 77,
#     "taylor_swift": 78,
#     "tom_cruise": 79,
#     "tom_hardy": 80,
#     "tom_hiddleston": 81,
#     "tom_holland": 82,
#     "tuppence_middleton": 83,
#     "ursula_corbero": 84,
#     "wentworth_miller": 85,
#     "zac_efron": 86,
#     "zendaya": 87,
#     "zoe_saldana": 88,
#     "alycia_dabnem_carey": 89,
#     "amber_heard": 90,
#     "barack_obama": 91,
#     "barbara_palvin": 92,
#     "camila_mendes": 93,
#     "elizabeth_olsen": 94,
#     "ellen_page": 95,
#     "elon_musk": 96,
#     "gal_gadot": 97,
#     "grant_gustin": 98,
#     "jeff_bezos": 99,
#     "kiernen_shipka": 100,
#     "margot_robbie": 101,
#     "melissa_fumero": 102,
#     "scarlett_johansson": 103,
#     "tom_ellis": 104,
# }

# Caltech101 class mappings for unlearning experiments
# Note: Despite the name "Caltech101", this dataset actually has 102 valid classes (labels 0-101).
# The original dataset has 103 folders, but torchvision removes BACKGROUND_Google, leaving 102 categories.
# Categories are sorted with capital letters first (Faces, Faces_easy, Leopards, Motorbikes), then lowercase.
caltech101_dict = {
    "Faces": 0,
    "Faces_easy": 1,
    "Leopards": 2,
    "Motorbikes": 3,
    "accordion": 4,
    "airplanes": 5,
    "anchor": 6,
    "ant": 7,
    "barrel": 8,
    "bass": 9,
    "beaver": 10,
    "binocular": 11,
    "bonsai": 12,
    "brain": 13,
    "brontosaurus": 14,
    "buddha": 15,
    "butterfly": 16,
    "caltech101": 17,  # meta-category - not typically used for unlearning
    "camera": 18,
    "cannon": 19,
    "car_side": 20,
    "ceiling_fan": 21,
    "cellphone": 22,
    "chair": 23,
    "chandelier": 24,
    "cougar_body": 25,
    "cougar_face": 26,
    "crab": 27,
    "crayfish": 28,
    "crocodile": 29,
    "crocodile_head": 30,
    "cup": 31,
    "dalmatian": 32,
    "dollar_bill": 33,
    "dolphin": 34,
    "dragonfly": 35,
    "electric_guitar": 36,
    "elephant": 37,
    "emu": 38,
    "euphonium": 39,
    "ewer": 40,
    "ferry": 41,
    "flamingo": 42,
    "flamingo_head": 43,
    "garfield": 44,
    "gerenuk": 45,
    "gramophone": 46,
    "grand_piano": 47,
    "hawksbill": 48,
    "headphone": 49,
    "hedgehog": 50,
    "helicopter": 51,
    "ibis": 52,
    "inline_skate": 53,
    "joshua_tree": 54,
    "kangaroo": 55,
    "ketch": 56,
    "lamp": 57,
    "laptop": 58,
    "llama": 59,
    "lobster": 60,
    "lotus": 61,
    "mandolin": 62,
    "mayfly": 63,
    "menorah": 64,
    "metronome": 65,
    "minaret": 66,
    "nautilus": 67,
    "octopus": 68,
    "okapi": 69,
    "pagoda": 70,
    "panda": 71,
    "pigeon": 72,
    "pizza": 73,
    "platypus": 74,
    "pyramid": 75,
    "revolver": 76,
    "rhino": 77,
    "rooster": 78,
    "saxophone": 79,
    "schooner": 80,
    "scissors": 81,
    "scorpion": 82,
    "sea_horse": 83,
    "snoopy": 84,
    "soccer_ball": 85,
    "stapler": 86,
    "starfish": 87,
    "stegosaurus": 88,
    "stop_sign": 89,
    "strawberry": 90,
    "sunflower": 91,
    "tick": 92,
    "trilobite": 93,
    "umbrella": 94,
    "watch": 95,
    "water_lilly": 96,
    "wheelchair": 97,
    "wild_cat": 98,
    "windsor_chair": 99,
    "wrench": 100,
    "yin_yang": 101,
}

# ==============================================================================
# ### EVALUATION METRICS ###
# ==============================================================================

# All valid evaluation metric names. Files live flat under supreme/eval_metrics/.
evaluation_metrics = [
    "accuracy",
    "activation_distance",
    "completeness",
    "jsdiv",
    "layerwise_distance",
    "membership_inference_attack",
    "resource_consumption",
    "time",
    "zrf",
]

# Subset that requires a retrained reference model (M_r). The retrain pipeline
# is only triggered when at least one requested metric is in this set.
metrics_requiring_retrain = {
    "activation_distance",
    "completeness",
    "jsdiv",
    "layerwise_distance",
    "membership_inference_attack",
    "time",
}

# ==============================================================================
# ### TRAINING HYPERPARAMETERS ###
# ==============================================================================
# Total training epochs and milestones (when learning rate gets lowered)

################
### ResNet18 ###
################
Cifar10_RN_EPOCHS = 20
Cifar10_RN_MILESTONES = [8, 12, 16]

Cifar20_RN_EPOCHS = 40
Cifar20_RN_MILESTONES = [15, 30, 35]

Cifar100_RN_EPOCHS = 200
Cifar100_RN_MILESTONES = [60, 120, 160]

PinsFaceRecognition_RN_EPOCHS = 200
PinsFaceRecognition_RN_MILESTONES = [60, 120, 160]

Caltech101_RN_EPOCHS = 100
Caltech101_RN_MILESTONES = [30, 60, 80]


################
###### ViT #####
################
# ViT uses AdamW + CosineAnnealingLR (milestones kept for backward compat but unused)
ViT_LR = 5e-5

Cifar10_ViT_EPOCHS = 8
Cifar10_ViT_MILESTONES = [7]

Cifar20_ViT_EPOCHS = 9
Cifar20_ViT_MILESTONES = [8]

Cifar100_ViT_EPOCHS = 8
Cifar100_ViT_MILESTONES = [7]

PinsFaceRecognition_ViT_EPOCHS = 8
PinsFaceRecognition_ViT_MILESTONES = [7]

Caltech101_ViT_EPOCHS = 8
Caltech101_ViT_MILESTONES = [7]

# ==============================================================================
# ### OTHER CONFIGURATIONS ###
# ==============================================================================
# Project root (app/host). Defaults to the repository root inferred from this
# file's location, which is correct for source checkouts and editable installs
# (pip install -e .). When SUPREME is installed as a wheel into site-packages
# and reused from another project, set SUPREME_PROJECT_ROOT to point logs/ and
# checkpoints at a writable working directory. The default is unchanged, so the
# paper's reproduction behaviour is unaffected.
# This file is at src/supreme/utils/, so the repo root is three directories up
# (utils -> supreme -> src -> repo root).
PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "SUPREME_PROJECT_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    )
)  # Get the absolute path to the project root (app/host)
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "logs",
    "training",
)  # Define checkpoint path relative to project root

DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
TIME_NOW = datetime.now().strftime(DATE_FORMAT)  # time of script run


# ==============================================================================
# ### HELPER FUNCTIONS ###
# ==============================================================================


# ==============================================================================
# ### PATH CONSTRUCTION FUNCTIONS ###
# ==============================================================================
# Centralized path construction ensures consistency across the codebase.
# All paths follow the structure:
#   logs/{phase}/precision_{precision}/{scenario}/{num_gpus}gpus/{seed_component}/...


def get_base_log_path(
    phase: str,
    precision: str,
    num_gpus: int,
    include_gpus_in_path: bool = True,
) -> str:
    """
    Construct the base log path with consistent structure.

    Args:
        phase: "training" or "unlearning"
        precision: Precision string (e.g., "32-true")
        num_gpus: Number of GPUs used
        include_gpus_in_path: Whether to include GPU count in path

    Returns:
        str: Base path like "logs/training/precision_32-true/4gpus/"

    Examples:
        >>> get_base_log_path("unlearning", "32-true", 4)
        'logs/unlearning/precision_32-true/4gpus'
        >>> get_base_log_path("training", "bf16-mixed", 1)
        'logs/training/precision_bf16-mixed/1gpus'
    """
    components = [PROJECT_ROOT, "logs", phase, f"precision_{precision}"]

    if include_gpus_in_path:
        components.append(f"{num_gpus}gpus")

    return os.path.join(*components)


def get_unlearning_dataset_path(
    precision: str,
    num_gpus: int,
    seed: int,
    strategy: str,
    dataset_name: str,
    model_name: str,
    include_gpus_in_path: bool = True,
) -> str:
    """
    Construct the path for unlearning processed datasets (trainset.pt, testset.pt).

    This path is used for caching original datasets during unlearning.
    Structure: logs/unlearning/precision_{precision}/{num_gpus}gpus/seed_{seed}/{strategy}/{dataset}/{model}/

    Args:
        precision: Precision string (e.g., "32-true")
        num_gpus: Number of GPUs used
        seed: Unlearning seed
        strategy: Unlearning strategy (fullclass, subclass, random_)
        dataset_name: Dataset name (e.g., "Cifar10")
        model_name: Model name (e.g., "ResNet18")
        include_gpus_in_path: Whether to include GPU count in path

    Returns:
        str: Full path to dataset directory

    Example:
        >>> get_unlearning_dataset_path("32-true", 4, 60, "random_", "Cifar10", "ResNet18")
        '.../logs/unlearning/precision_32-true/4gpus/seed_60/random_/Cifar10/ResNet18'
    """
    base_path = get_base_log_path(
        phase="unlearning",
        precision=precision,
        num_gpus=num_gpus,
        include_gpus_in_path=include_gpus_in_path,
    )

    return os.path.join(base_path, f"seed_{seed}", strategy, dataset_name, model_name)


def get_training_dataset_path(
    precision: str,
    seed: int,
    strategy: str,
    dataset_name: str,
    model_name: str,
) -> str:
    """
    Construct the path for training processed datasets.

    Training datasets don't include GPU count since the data is shared across configurations.
    Structure: logs/training/precision_{precision}/processed_datasets/seed_{seed}/{strategy}/{dataset}/{model}/

    Args:
        precision: Precision string (e.g., "32-true")
        seed: Training seed
        strategy: Unlearning strategy (can be None for training-only scenarios)
        dataset_name: Dataset name (e.g., "Cifar10")
        model_name: Model name (e.g., "ResNet18")

    Returns:
        str: Full path to dataset directory
    """
    base_path = os.path.join(
        PROJECT_ROOT, "logs", "training", f"precision_{precision}", "processed_datasets"
    )

    components = [base_path, f"seed_{seed}"]
    if strategy:
        components.append(strategy)
    components.extend([dataset_name, model_name])

    return os.path.join(*components)


def get_dataset_path_from_log_dir(log_dir: str, model_name: str) -> str:
    """
    Derive the dataset path from LOG_DIR environment variable.

    LOG_DIR structure: .../precision_xxx/Ngpus/seed_xxx/strategy/dataset/net/classes_xxx/forget_xxx/
    Dataset path (3 directories up): .../precision_xxx/Ngpus/seed_xxx/strategy/dataset/

    Args:
        log_dir: The LOG_DIR value (full path to forget target directory)
        model_name: Model name to append

    Returns:
        str: Path to dataset directory (for trainset.pt, testset.pt)

    Example:
        >>> get_dataset_path_from_log_dir(".../seed_60/random_/Cifar10/ResNet18/classes_10/forget_perc_0.01", "ResNet18")
        '.../seed_60/random_/Cifar10/ResNet18'
    """
    # Go up 3 directories from LOG_DIR to get dataset base
    dataset_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(log_dir)))
    return os.path.join(dataset_base_dir, model_name)


def get_dict_name_for_dataset(dataset: str, strategy: str) -> str:
    """
    Get the appropriate class dictionary name for a dataset/strategy combination.

    This centralizes the logic for determining which class dictionary to use,
    avoiding duplication across bash scripts and Python code.

    Args:
        dataset: Dataset name (e.g., "Cifar20", "Cifar100", "PinsFaceRecognition", "Caltech101")
        strategy: Unlearning strategy (e.g., "fullclass", "subclass", "random_")

    Returns:
        str: Dictionary variable name (e.g., "cifar20_dict", "cifar100_dict", "pins_dict")

    Raises:
        ValueError: If dataset/strategy combination is unknown

    Examples:
        >>> get_dict_name_for_dataset("Cifar20", "fullclass")
        'cifar20_dict'
        >>> get_dict_name_for_dataset("Cifar20", "subclass")
        'cifar100_dict'  # Special case: subclass uses CIFAR100 fine-grained classes
    """
    # Special case: Cifar20 subclass uses CIFAR100 classes for fine-grained unlearning
    if dataset == "Cifar20" and strategy == "subclass":
        return "cifar100_dict"

    # Map datasets to their dictionary variable names
    dataset_to_dict = {
        "Cifar20": "cifar20_dict",
        "Cifar100": "cifar100_dict",
        "PinsFaceRecognition": "pins_dict",
        "Caltech101": "caltech101_dict",
    }

    if dataset not in dataset_to_dict:
        # Externally registered datasets store their class dict on this module
        # under "<dataset>_dict" (see supreme.registry.register_dataset).
        from supreme.registry import get_external_dataset_dict_name

        external = get_external_dataset_dict_name(dataset)
        if external is not None:
            return external

        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported datasets: {', '.join(dataset_to_dict.keys())}"
        )

    return dataset_to_dict[dataset]
