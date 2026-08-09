# PlantGuard — Crop Disease Prediction App

A deep-learning web app that identifies plant leaf diseases from a photo. Upload a leaf image and a
convolutional neural network classifies it into one of **38 classes** (healthy + diseased) across 14
crop species.

The project has two parts:

- **Training** — [newplantclassification.ipynb](newplantclassification.ipynb), a Keras/TensorFlow
  notebook that builds, trains and evaluates the CNN on the PlantVillage dataset.
- **Serving** — a [Streamlit](https://streamlit.io) app ([plantpredv2.py](plantpredv2.py)) with a
  Home / About / Disease Recognition dashboard that loads the trained `.keras` model and predicts on
  uploaded images.

---

## Table of contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The models](#the-models)
- [Dataset](#dataset)
- [Results](#results)
- [Supported classes](#supported-classes)
- [Retraining](#retraining)
- [GitHub Codespaces / Dev Container](#github-codespaces--dev-container)
- [Known issues and limitations](#known-issues-and-limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Quickstart

Requires Python 3.10–3.12 (the notebook was run on 3.12.4). TensorFlow does not yet publish wheels
for very new Python releases, so avoid 3.13+.

```bash
git clone https://github.com/Gbolahan43/crop-disease-app.git
cd crop-disease-app

python -m venv .venv
# Windows (bash):  source .venv/Scripts/activate
# macOS / Linux:   source .venv/bin/activate
source .venv/bin/activate

pip install -r requirements.txt
```

Run whichever front-end you prefer — all three share the same model and class registry:

```bash
streamlit run plantpredv2.py     # dark theme, single prediction + confidence
streamlit run newplantpred.py    # light "field report", top-3 ranked candidates
streamlit run plantpred.py       # original minimal v1 UI
```

Then open http://localhost:8501, pick **Disease Recognition** in the sidebar, upload a `.jpg`/`.jpeg`/
`.png` leaf image and run the prediction.

All paths are resolved relative to the scripts, so they can be launched from any working directory.

The pretrained weights (`newplantdis.keras`, `trainedv3.keras`) are committed directly to the repo —
about 60 MB each, so the initial clone is ~120 MB.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| [plant_classes.py](plant_classes.py) | **Shared registry.** The 38 class labels, model/asset paths, input size and label-formatting helpers. Imported by all three apps so the label order can never drift between them. |
| [plantpredv2.py](plantpredv2.py) | **Dark theme.** Custom CSS, cards, radio nav, confidence score with a low-confidence warning. Launched by the dev container. |
| [newplantpred.py](newplantpred.py) | **Light "field report" theme.** Redesigned: paper-like layout, **top-3 ranked candidates** with confidence bars, coverage listed from the shared registry. |
| [plantpred.py](plantpred.py) | **v1 reference.** The original minimal UI — plain widgets, no CSS, single prediction. Deliberately left plain; only correctness fixes applied. |
| [newplantclassification.ipynb](newplantclassification.ipynb) | Training notebook: data loading, two model definitions, training, evaluation, confusion matrix, single-image inference. |
| [newplantdis.keras](newplantdis.keras) | Trained weights for `model2` (5 epochs) — the better-performing model. |
| [trainedv3.keras](trainedv3.keras) | Trained weights for `model` (10 epochs). The notebook cell saves this one as `trainedv3.h5`; the committed `.keras` file was exported outside the recorded cells. |
| [training_histv3.json](training_histv3.json) | Per-epoch accuracy/loss history for `model` (10 epochs). |
| [training_hist2v3.json](training_hist2v3.json) | Per-epoch accuracy/loss history for `model2` (5 epochs). |
| [home_page.jpg](home_page.jpg) | Hero image shown on the app's Home page. |
| [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) | Codespaces / Dev Container setup that auto-launches Streamlit on port 8501. |
| `requirements.txt` | Python dependencies, floor- and ceiling-pinned. |
| [.gitignore](.gitignore) | Ignores the dataset directories, notebook-generated figures, checkpoints other than the committed `.keras` files, and the usual Python/Jupyter/Streamlit/OS noise. |

**All three front-ends load `newplantdis.keras`**, the more accurate checkpoint, via
`BEST_MODEL_PATH` in [plant_classes.py](plant_classes.py) — change it in one place to switch every
app. They share the class list and path constants from that module, so none of them redefines the 38
labels.

---

## The models

Both `model` and `model2` in the notebook use the **same architecture** — a VGG-style CNN built with
`tf.keras.Sequential`:

```
Rescaling(1/255)                      input 128 x 128 x 3
5 x [ Conv2D(same) -> BatchNorm -> Conv2D(valid) -> BatchNorm -> pool ]
      filters: 32, 64, 128, 256, 512
      MaxPooling2D(2) after blocks 1-4; GlobalAveragePooling2D after block 5
Flatten -> Dense(512, relu) -> Dropout(0.5) -> Dense(38, softmax)
```

- **Total parameters:** 5,002,310 (~19 MB) — 4,998,342 trainable
- **Optimizer:** Adam, learning rate `1e-4`
- **Loss:** `categorical_crossentropy` (labels are one-hot, `label_mode="categorical"`)
- **Batch size:** 32
- **Input size:** images resized to 128×128, bilinear interpolation
- **Epochs:** `model` = 10, `model2` = 5

No pretrained backbone and no online augmentation are used — the dataset itself is pre-augmented.

Inference in the app mirrors training: load the image at `target_size=(128, 128)`, convert to an
array, add a batch axis, `predict`, then `argmax` over the 38 logits. Rescaling lives *inside* the
model, so raw 0–255 pixel values are the correct input — do not divide by 255 before predicting.

---

## Dataset

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), in the widely used
offline-augmented variant (~87,000 RGB leaf images, 38 classes).

As loaded in the notebook via `image_dataset_from_directory`:

| Split | Directory | Images | Classes |
| --- | --- | --- | --- |
| Train | `train/` | 70,295 | 38 |
| Validation | `valid/` | 17,572 | 38 |
| Ad-hoc test | `test/` | ~33 images used for spot-check predictions | — |

Expected layout (one subdirectory per class; labels are inferred from folder names):

```
train/
  Apple___Apple_scab/
  Apple___Black_rot/
  ...
valid/
  Apple___Apple_scab/
  ...
test/
  PotatoHealthy2.JPG
  ...
```

The image folders are **not** committed to this repo — download the dataset and place `train/`,
`valid/` and `test/` next to the notebook before retraining.

---

## Results

Evaluated with `model.evaluate()` and `sklearn.metrics.classification_report` on the 17,572-image
`valid/` split:

| Checkpoint | Notebook variable | Epochs | Train accuracy | Validation accuracy | Validation loss |
| --- | --- | --- | --- | --- | --- |
| `newplantdis.keras` | `model2` | 5 | 0.9894 | **0.9767** | 0.0736 |
| `trainedv3.keras` | `model` | 10 | 0.9042 | 0.8860 | 0.4288 |

Per-class report for `model2` (`newplantdis.keras`): **0.98 accuracy, 0.98 macro-average precision,
recall and F1** across all 38 classes. Weakest classes are `Tomato___Late_blight` (recall 0.86) and
`Potato___Late_blight` (precision 0.90) — late blight is visually confusable between hosts.

Two caveats on the headline "98%":

1. It is measured on the **validation** split, which was also used for monitoring during training —
   there is no untouched held-out test set. Treat it as an optimistic estimate.
2. `trainedv3.keras` (the checkpoint the current app loads) is the **weaker** model. Its 10-epoch run
   spiked on the final epoch — validation accuracy fell from 0.9824 at epoch 8 to 0.8860 at epoch 10
   while validation loss rose from 0.055 to 0.429, and the weights were saved after that regression.
   For best accuracy, run [newplantpred.py](newplantpred.py) instead, which loads `newplantdis.keras`.

Per-epoch curves are in [training_histv3.json](training_histv3.json) and
[training_hist2v3.json](training_hist2v3.json); the notebook plots accuracy/loss and a 38×38
confusion matrix heatmap.

---

## Supported classes

38 labels in `Crop___Condition` form, in the exact index order the model outputs:

| Crop | Classes |
| --- | --- |
| Apple | Apple scab, Black rot, Cedar apple rust, healthy |
| Blueberry | healthy |
| Cherry (incl. sour) | Powdery mildew, healthy |
| Corn (maize) | Cercospora / Gray leaf spot, Common rust, Northern leaf blight, healthy |
| Grape | Black rot, Esca (Black Measles), Leaf blight (Isariopsis), healthy |
| Orange | Huanglongbing (citrus greening) |
| Peach | Bacterial spot, healthy |
| Pepper (bell) | Bacterial spot, healthy |
| Potato | Early blight, Late blight, healthy |
| Raspberry | healthy |
| Soybean | healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellow leaf curl virus, Mosaic virus, healthy |

The class list is hard-coded in each app script and **must stay in the alphabetical order that
`image_dataset_from_directory` infers** — reordering it silently mislabels every prediction.

### Tips for good predictions

- Good, even lighting; avoid harsh shadows and glare.
- Fill the frame with a single leaf, focused on the affected area.
- Plain background where possible.
- The model only knows the 14 crops above. Anything else (a different species, a whole plant, a
  non-leaf photo) still returns one of the 38 labels with high confidence — it cannot say "unknown".

---

## Retraining

1. Download and extract the augmented PlantVillage dataset so `train/`, `valid/` and `test/` sit
   beside the notebook.
2. Install the training extras:

   ```bash
   pip install tensorflow matplotlib pandas seaborn scikit-learn pillow opencv-python jupyter
   ```

3. Open the notebook and run top to bottom:

   ```bash
   jupyter lab newplantclassification.ipynb
   ```

   It builds the datasets, defines `model`/`model2`, trains, saves checkpoints, dumps the histories to
   JSON, and produces the classification report and confusion matrix.
4. Point the app at your new checkpoint by editing the `load_model()` call in
   [plantpredv2.py:14](plantpredv2.py#L14).

Training on CPU is slow — a single `evaluate()` pass over the training set took ~18 minutes
(482 ms/step × 2,197 steps) on the original run. Use a GPU where available.

`ReduceLROnPlateau` / `EarlyStopping` are imported in the notebook but commented out of the `fit()`
calls; enabling them would likely have avoided the epoch-10 regression noted above.

---

## GitHub Codespaces / Dev Container

[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) provisions a Python 3.11 container
that installs `requirements.txt` plus Streamlit, forwards port 8501, and auto-runs the app on attach:

```bash
streamlit run plantpredv2.py --server.enableCORS false --server.enableXsrfProtection false
```

Change `postAttachCommand` to run a different front-end. The `enableCORS` / `enableXsrfProtection`
flags are there to make the forwarded-port preview work and should not be used in a real deployment.

---

## Known issues and limitations

Limits that are inherent to the model and dataset rather than bugs:

- **The accuracy figure is optimistic.** 97.7 % is measured on the validation split, which was also
  used to monitor training. There is no untouched test set.
- **Lab conditions, not field conditions.** PlantVillage images are single leaves under even light on
  a uniform background. Real field photography will score materially worse. The apps mitigate this
  with a 60 % confidence floor, but cannot eliminate it.
- **Closed-world classifier.** Softmax over 38 classes always names one of them. An off-domain photo
  (an unsupported crop, a whole plant, a non-leaf subject) still gets a label — the confidence floor
  flags the likely cases, but a confident wrong answer is possible.
- **No treatment guidance.** The apps report a label and confidence, and point you at an agronomist.
  They do not prescribe pesticide or dosage.
- **Not a diagnostic authority.** Do not use this as the sole basis for pesticide or crop-management
  decisions.
- **Large binaries in Git.** Two ~60 MB `.keras` files are committed without Git LFS, so every clone
  and every future checkpoint bloats history.
- **`trainedv3.keras` is kept but unused** — no app loads it. It is retained for comparison against
  `newplantdis.keras`; see [Results](#results).

---

## Roadmap

- Retrain with `EarlyStopping` + `ReduceLROnPlateau` and a `ModelCheckpoint` on best `val_loss`; hold
  out a true test split so the accuracy figure is honest.
- Try transfer learning (EfficientNet / MobileNetV3) for better field-photo generalisation and a
  smaller model.
- Show top-3 candidates in `plantpredv2.py` too, as `newplantpred.py` now does.
- Add real per-disease treatment guidance behind the label.
- Calibrate the 60 % confidence floor against measured out-of-domain photos rather than picking it by
  eye.
- Add a smoke test in CI that imports each app with `streamlit`/`tensorflow` stubbed, so API removals
  are caught before release.
- Move checkpoints to Git LFS or a release artifact.

---

## Acknowledgements

- Dataset: [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) by
  Sharada P. Mohanty et al.
- Built with [TensorFlow/Keras](https://www.tensorflow.org) and [Streamlit](https://streamlit.io).

Author: [Gbolahan43](https://github.com/Gbolahan43) · Repo:
[crop-disease-app](https://github.com/Gbolahan43/crop-disease-app)
