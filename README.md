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
pip install "streamlit<1.18"          # see Known issues below
```

Run the app:

```bash
streamlit run plantpredv2.py
```

Then open http://localhost:8501, pick **Disease Recognition** in the sidebar, upload a leaf image and
press **Predict**.

> `requirements.txt` currently pins only `tensorflow`. Streamlit is installed separately (the dev
> container does this in `updateContentCommand`). See [Known issues](#known-issues-and-limitations).

The pretrained weights (`newplantdis.keras`, `trainedv3.keras`) are committed directly to the repo —
about 60 MB each, so the initial clone is ~120 MB.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| [plantpredv2.py](plantpredv2.py) | **Current app.** Streamlit UI + error handling/logging; loads `trainedv3.keras`. |
| [newplantpred.py](newplantpred.py) | Same app, but loads `newplantdis.keras`. Identical to `plantpredv2.py` apart from that one line. |
| [plantpred.py](plantpred.py) | Original prototype app (no error handling). Referenced by the dev container. |
| [newplantclassification.ipynb](newplantclassification.ipynb) | Training notebook: data loading, two model definitions, training, evaluation, confusion matrix, single-image inference. |
| [newplantdis.keras](newplantdis.keras) | Trained weights for `model2` (5 epochs) — the better-performing model. |
| [trainedv3.keras](trainedv3.keras) | Trained weights for `model` (10 epochs). The notebook cell saves this one as `trainedv3.h5`; the committed `.keras` file was exported outside the recorded cells. |
| [training_histv3.json](training_histv3.json) | Per-epoch accuracy/loss history for `model` (10 epochs). |
| [training_hist2v3.json](training_hist2v3.json) | Per-epoch accuracy/loss history for `model2` (5 epochs). |
| [home_page.jpg](home_page.jpg) | Hero image shown on the app's Home page. |
| [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) | Codespaces / Dev Container setup that auto-launches Streamlit on port 8501. |
| `requirements.txt` | Python dependencies. |

The three `*pred*.py` files are near-duplicates that differ only in which checkpoint they load — a
consolidation candidate (see [Roadmap](#roadmap)).

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
streamlit run plantpred.py --server.enableCORS false --server.enableXsrfProtection false
```

Note it launches the **original** [plantpred.py](plantpred.py), not `plantpredv2.py`. Update
`postAttachCommand` if you want the current app. The `enableCORS` / `enableXsrfProtection` flags are
there to make the forwarded-port preview work and should not be used in a real deployment.

---

## Known issues and limitations

These are real, currently-open problems in the code — worth knowing before you file a bug:

- **Streamlit caching API is obsolete.** All three app scripts use
  `@st.cache(allow_output_mutation=True)`, which was deprecated in Streamlit 1.18 and **removed** in
  later versions; the app raises on import with modern Streamlit. Fix by switching to the
  `@st.cache_resource` line already present but commented out just below it (e.g.
  [plantpredv2.py:10-11](plantpredv2.py#L10-L11)).
- **`use_column_width` is deprecated** in current Streamlit in favour of `use_container_width`.
- **`plantpred.py` crashes on an empty upload.** Its `Show Image` / `Predict` buttons are outside any
  `if test_image is not None` guard, so clicking them before uploading passes `None` downstream.
  `plantpredv2.py` and `newplantpred.py` fix this.
- **The default app loads the weaker checkpoint** (`trainedv3.keras`). See
  [Results](#results).
- **`requirements.txt` is incomplete** — it lists only `tensorflow`, with no `streamlit`, no `numpy`
  and no version pins, so a fresh environment is not reproducible.
- **Duplicated app code.** `plantpred.py`, `newplantpred.py` and `plantpredv2.py` are near-identical;
  the 38-class list is copy-pasted into each.
- **Typo:** "sIdentify" on the Disease Recognition page of [plantpred.py:109](plantpred.py#L109).
- **UI overstates the product.** The Home/About copy promises "treatment recommendations", a
  "comprehensive database" and a "continually updated" model. None of that is implemented — the app
  returns a single class label and nothing else.
- **No confidence score is shown.** `argmax` discards the softmax probability, so a 35 % guess and a
  99 % match look identical to the user. Surfacing `np.max(prediction)` would be a cheap improvement.
- **Not a diagnostic authority.** Trained on lab-style, single-leaf, uniform-background images;
  accuracy on real-world field photos will be markedly lower. Do not use it as the sole basis for
  pesticide or crop-management decisions.
- **Large binaries in Git.** Two ~60 MB `.keras` files are committed without Git LFS, so every clone
  and every future checkpoint bloats history.

---

## Roadmap

- Migrate to `@st.cache_resource` and `use_container_width`; pin dependencies in `requirements.txt`.
- Collapse the three app scripts into one, with the model path and class list in a shared module or
  config.
- Display top-k predictions with confidence scores, and an "uncertain" threshold.
- Retrain with `EarlyStopping` + `ReduceLROnPlateau` and a `ModelCheckpoint` on best `val_loss`; hold
  out a true test split.
- Try transfer learning (EfficientNet / MobileNetV3) for better field-photo generalisation and a
  smaller model.
- Add the treatment-recommendation content the UI already advertises, or trim the claims.
- Move checkpoints to Git LFS or a release artifact.

---

## Acknowledgements

- Dataset: [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) by
  Sharada P. Mohanty et al.
- Built with [TensorFlow/Keras](https://www.tensorflow.org) and [Streamlit](https://streamlit.io).

Author: [Gbolahan43](https://github.com/Gbolahan43) · Repo:
[crop-disease-app](https://github.com/Gbolahan43/crop-disease-app)
