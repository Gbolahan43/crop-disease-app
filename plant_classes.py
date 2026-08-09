"""Shared constants and helpers for the PlantGuard apps.

Single source of truth for the 38 PlantVillage class labels, the model/asset
paths and the label-formatting helpers used by every front-end in this repo
(``plantpred.py``, ``newplantpred.py``, ``plantpredv2.py``).

The label order MUST match the alphabetical order that
``tf.keras.utils.image_dataset_from_directory`` infers from the dataset folder
names. The models emit an index into this list, so reordering it silently
mislabels every prediction.

Paths are resolved relative to this file so the apps work no matter which
directory Streamlit is launched from.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trained as `model2` in newplantclassification.ipynb (5 epochs).
# Best of the two checkpoints: val accuracy 0.9767, val loss 0.0736,
# macro-average precision/recall/F1 0.98 over the 17,572-image valid split.
BEST_MODEL_PATH = os.path.join(BASE_DIR, "newplantdis.keras")

# Trained as `model` (10 epochs). Regressed on the final epoch -- val accuracy
# fell 0.9824 (epoch 8) -> 0.8860 (epoch 10) while val loss rose 0.055 -> 0.429
# -- and the weights were saved after that regression. Kept for comparison;
# prefer BEST_MODEL_PATH.
V3_MODEL_PATH = os.path.join(BASE_DIR, "trainedv3.keras")

HOME_IMAGE_PATH = os.path.join(BASE_DIR, "home_page.jpg")

# Both models were trained on 128x128 RGB crops with Rescaling(1/255) as the
# first layer, so raw 0-255 pixel values are the correct model input.
IMAGE_SIZE = (128, 128)

# File types the uploaders accept.
ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# The dataset folder misspells huanglongbing; the label must stay as-is to match
# the model's output index, but it can be displayed correctly.
_DISPLAY_FIXES = {
    "Haunglongbing (Citrus greening)": "Huanglongbing (citrus greening)",
}


def split_label(label):
    """Split a raw label into ``(crop, condition)`` display strings.

    >>> split_label("Pepper,_bell___Bacterial_spot")
    ('Pepper, bell', 'Bacterial spot')
    """
    crop, _, condition = label.partition("___")
    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return crop, _DISPLAY_FIXES.get(condition, condition)


def crop_name(label):
    """Human-readable crop for a raw label, e.g. ``'Corn (maize)'``."""
    return split_label(label)[0]


def condition_name(label):
    """Human-readable condition for a raw label, e.g. ``'Bacterial spot'``."""
    return split_label(label)[1]


def pretty_label(label):
    """Full human-readable label, e.g. ``'Tomato - Late blight'``."""
    crop, condition = split_label(label)
    return f"{crop} — {condition}" if condition else crop


def is_healthy(label):
    """True when the label denotes a healthy leaf rather than a disease."""
    return condition_name(label).lower() == "healthy"


def crops():
    """Sorted list of the distinct crops the model covers."""
    return sorted({crop_name(label) for label in CLASS_NAMES})


def diseases_by_crop():
    """Map each crop to its list of conditions, in model-index order."""
    grouped = {}
    for label in CLASS_NAMES:
        crop, condition = split_label(label)
        grouped.setdefault(crop, []).append(condition)
    return grouped
