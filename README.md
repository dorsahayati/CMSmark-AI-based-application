# CMSmark

CMSmark is a desktop application for classifying gene-expression samples into
Consensus Molecular Subtype (CMS) classes using pre-trained machine-learning
models. It provides a graphical interface for uploading count-based
expression data, running inference with four independently trained
classifiers, and generating per-model and cross-model comparison results.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Machine-Learning Models](#3-machine-learning-models)
- [4. Feature Selection](#4-feature-selection)
- [5. Installation](#5-installation)
- [6. Running the Application](#6-running-the-application)
- [7. Input Data](#7-input-data)
- [8. Prediction Workflow](#8-prediction-workflow)
- [9. Results Structure](#9-results-structure)
- [10. Result Interpretation](#10-result-interpretation)
- [11. Reproducibility](#11-reproducibility)
- [12. Citation](#12-citation)
- [13. License](#13-license)
- [14. Authors / Contact](#14-authors--contact)

## 1. Overview

**What CMSmark does.** CMSmark takes a gene-expression count matrix as input
and predicts, for each sample, a class label (`CMS1`, `CMS2`, `CMS3`,
`CMS4`, or `Unclassified` when no class reaches the confidence threshold).
Predictions are produced independently by four models and are reported both
per model and as a combined, cross-model comparison.

**Prediction workflow.** A raw count matrix is normalized to
counts-per-million (CPM), optionally followed by a log2(CPM + 1) transform,
and the resulting expression values for a fixed panel of genes are passed to
each of the four trained models. Each model outputs a predicted class and a
confidence score; predictions below a user-defined probability threshold are
labeled `Unclassified`.

**Role of the four models.** Logistic Regression, Random Forest, XGBoost,
and a Multi-Layer Perceptron (MLP) are trained independently on the same
gene panel and are run side by side during inference. Reporting all four
outputs — plus their agreement with one another — allows a reviewer to
assess how consistent the classification is across different modeling
approaches, rather than relying on a single model's prediction.

**Research context.** This software is the inference and reporting tool
used for the analyses described in the associated manuscript. Model
training, dataset curation, and biological interpretation are outside the
scope of this repository; only the trained model artifacts required for
inference are included.

## 2. Repository Structure

```text
CMSmark-AI-based-application/
├── build/                       # PyInstaller intermediate build output (not needed to run the app)
├── dist/
│   └── CMSMARK.exe              # Pre-built, ready-to-run Windows executable — no Python required
├── icon/                        # Application icon assets
├── rthooks/                     # PyInstaller runtime hook (XGBoost VERSION-file fix)
├── src/
│   ├── gui/
│   │   └── app_window.py        # GUI window, controls, and event handling
│   ├── models/
│   │   ├── mlp_models/          # Trained MLP model
│   │   ├── rf_lr_models/        # Trained Logistic Regression and Random Forest models
│   │   └── xgb_models/          # Trained XGBoost model (PCA-reduced pipeline)
│   ├── utils/
│   │   └── file_utils.py        # Basic CSV read/write/validation helpers
│   ├── cmsmark.py                # Inference pipeline, plotting, and result generation
│   └── main.py                   # Application entry point (launches the GUI)
├── build.ps1                    # Windows build script (regenerates build/ and dist/)
├── CMSmark.spec                 # PyInstaller build specification (current build)
├── convert_icon.py              # Icon conversion helper (not used at runtime)
├── PythonGUIApp.spec            # Earlier/minimal PyInstaller specification
├── README.md
└── requirements.txt             # Python dependencies
```

`dist/CMSMARK.exe` is a pre-built, self-contained Windows executable —
reviewers or users who only want to run the classifier can download and
double-click it directly, without installing Python or any dependencies
(see [Running the Application](#6-running-the-application)). `build/`
holds PyInstaller's intermediate build artifacts from producing that
executable and is not needed to run the app.

Each model directory under `src/models/` contains the trained model file
(`.pkl`), its label encoder, its feature list (`features_*.txt`), and, where
used for visualization, a scaler (`scaler_*.pkl`).

## 3. Machine-Learning Models

Four models are trained on the same gene panel and used together during
inference:

- **Logistic Regression** — a linear classifier, included as an
  interpretable baseline.
- **Random Forest** — an ensemble of decision trees, included as a
  non-linear baseline robust to feature interactions.
- **XGBoost** — a gradient-boosted tree ensemble. In this repository, the
  XGBoost model's pipeline additionally reduces the input genes to 100
  components via an internal PCA step before classification; this happens
  inside the model's own pipeline, so it still accepts the same raw
  gene-level input as the other three models.
- **Multi-Layer Perceptron (MLP)** — a feed-forward neural network
  classifier.

All four models were trained externally; this repository contains only the
serialized, trained artifacts used for inference (see
`src/models/`), not the training code or training data. No performance
metrics (accuracy, F1-score, etc.) are stored in this repository, so none
are reported here; performance figures, if reported, appear in the
associated manuscript.

## 4. Feature Selection

Inference in this application always uses a single, fixed panel of genes
referred to internally as the **Excellent** feature group — a set of 455
genes shared across all four models (`src/models/*/features_*.txt`). Other
feature groupings that exist as model artifacts in this repository (for
example `Good` and `Excellent-Good`) are not exposed in the GUI or used
during inference; only the `Excellent` panel is active.

This is enforced in code in `src/cmsmark.py`: the constant `FEATURE_GROUP`
is set to `"Excellent"` and is the only group referenced by the inference
routine (`perform_inference`). Each model's expected feature list is loaded
from its corresponding `features_*.txt` file, and the input data is checked
against that exact list before prediction — inference stops with an
explicit error if the input file is missing any required gene.

## 5. Installation

These steps are only needed to run the application from source or to
rebuild the executable. To just run the classifier on Windows, use the
prebuilt `dist\CMSMARK.exe` instead — see
[Running the Application](#6-running-the-application) — and skip this
section entirely.

The application is written in Python and uses PyQt5 for its interface.
Installation instructions below assume Windows with Conda, which is the
supported environment for building the packaged executable; the
application can also run from source on other platforms with a compatible
Python and Qt installation.

```bash
conda create -n cmsmark python=3.10
conda activate cmsmark
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. This is not explicitly pinned in the
repository, but it is the minimum version required by the
`scikit-learn==1.7.0` dependency listed in `requirements.txt`.

## 6. Running the Application

### Quick start: prebuilt Windows executable

For users who just want to run the classifier, `dist\CMSMARK.exe` is a
ready-to-run, self-contained executable — it bundles the trained models,
so no Python installation, virtual environment, or dependency setup is
required. Download or copy the file and double-click it to launch the
GUI on Windows.

### Running from source

To run from source instead (for development, or on non-Windows
platforms), from the repository root, with the environment from
[Installation](#5-installation) active:

```bash
python src/main.py
```

This launches the PyQt5 GUI. On Linux without an attached display (for
example, in a CI environment or WSL without an X server), run:

```bash
QT_QPA_PLATFORM=offscreen python src/main.py
```

### Rebuilding the executable

If the source code or models change, `dist\CMSMARK.exe` can be regenerated
with PyInstaller, driven by `CMSmark.spec`:

```powershell
.\build.ps1
```

This must be run on Windows (PyInstaller does not cross-compile). It
regenerates both `build\` (intermediate PyInstaller artifacts) and
`dist\CMSMARK.exe`, bundling the trained models so the resulting
executable can be copied to and run on another Windows machine without a
separate Python installation.

## 7. Input Data

The application expects a CSV file containing a gene-expression count
matrix, structured as:

- **Rows**: genes, identified in the first column (used as the row index).
- **Columns**: sample identifiers.
- **Values**: raw (or CPM-normalized) expression counts.

At minimum, the input file must contain all 455 genes listed in the
`Excellent` feature panel (`src/models/rf_lr_models/features_Excellent.txt`);
inference raises an explicit error if any required gene is missing.

Two run-time options are set in the GUI:

- **Threshold** — the minimum prediction probability (0.0–1.0, default
  0.5) required for a sample to receive a class label instead of
  `Unclassified`.
- **Apply Log2CPM+1 transformation** — a checkbox controlling whether the
  CPM-normalized values are additionally log2(x + 1) transformed before
  being passed to the models. If unchecked, CPM values are used directly.

Input files are selected through the GUI's file dialog; one or more CSV
files can be processed in a single run.

**Optional ground-truth labels.** If a file named
`<input_file_stem>_labels.csv` exists alongside the input CSV (sample name
in the first column, true class in the second), its values are read and
included as a `True Class` column in the combined all-model prediction
table. This file is optional and only affects that one output column.

## 8. Prediction Workflow

```text
Input CSV (genes x samples, raw counts)
   ↓
CPM normalization (+ optional log2(CPM+1))
   ↓
Excellent feature panel selection (455 genes)
   ↓
Logistic Regression / Random Forest / XGBoost / MLP
   ↓
Per-model predictions (class + confidence)
   ↓
Model-specific results (inference table, heatmaps)
   ↓
All-model comparison (combined table, agreement and confidence plots)
```

## 9. Results Structure

Results are written under a `results/` folder inside the application's own
directory (next to `src/` when running from source, or next to the built
executable), never next to the input file. Each input file produces its own
`<input_file_stem>_results/` folder, re-generated on every run of that
file:

```text
results/
└── <input_file_stem>_results/
    ├── normalized_log_cpm.csv / .html
    ├── logistic_regression/
    │   ├── inference_table.csv / .html
    │   ├── heatmap.png / .html
    │   └── heatmap_interactive.html
    ├── random_forest/
    │   └── ... (same layout)
    ├── mlp/
    │   └── ... (same layout)
    ├── xgboost/
    │   └── ... (same layout)
    └── all_models_comparison/
        ├── all_model_predictions.csv / .html
        ├── agreement_matrix.png / .html
        ├── classification_results.png / .html
        ├── confidence_boxplot.png / .html
        └── venn_diagram_model_agreement.png / .html
```

Each model's own folder contains only that model's output. Files that
compare all four models together are kept exclusively in
`all_models_comparison/`.

## 10. Result Interpretation

- **`inference_table.csv` / `.html`** — per-sample prediction and
  per-class probability for one model.
- **`heatmap.png` / `heatmap_interactive.html`** — Z-scored expression of
  the most variable genes (within that model's confidently classified
  samples), grouped by predicted class.
- **`all_model_predictions.csv` / `.html`** — one row per sample, one
  column per model (prediction and confidence), aligned side by side for
  direct comparison.
- **`agreement_matrix.png` / `.html`** — a 4x4 heatmap where each cell is
  the fraction of samples on which two models produced the same
  prediction.
- **`classification_results.png` / `.html`** — per-model class-distribution
  pie charts alongside an all-model agreement summary.
- **`confidence_boxplot.png` / `.html`** — distribution of each model's
  maximum prediction probability, relative to the classification
  threshold.
- **`venn_diagram_model_agreement.png` / `.html`** — a four-set Venn-style
  diagram, one panel per CMS class, showing exactly how many samples were
  assigned that class by each combination of the four models. Because
  matplotlib's Venn-diagram support is limited to two or three sets, this
  is drawn as a non-area-proportional four-ellipse diagram (every
  intersection region is exact and uniquely labeled, but region size is
  not meant to be read as proportional to sample count).

## 11. Reproducibility

- **Environment**: Python 3.10+ with the packages listed in
  `requirements.txt` (see [Installation](#5-installation)).
- **Model artifacts**: inference requires the trained model files, label
  encoders, and feature lists under `src/models/`, which are included in
  this repository. No external model download is required.
- **Running inference**: launch the application — either the prebuilt
  `dist\CMSMARK.exe` on Windows, or `python src/main.py` from source —
  upload an input CSV matching the format in
  [Input Data](#7-input-data), set the threshold and normalization option,
  and run inference.
- **Outputs**: generated files appear under `results/` as described in
  [Results Structure](#9-results-structure).

**Known limitation.** `requirements.txt` pins `scikit-learn==1.7.0`. The
bundled model files were serialized with that version; running inference
with a different scikit-learn version may produce version-mismatch
warnings from `pickle` when loading the models. Using the pinned version is
recommended for exact reproducibility. Additionally, this repository does
not include the raw expression datasets used to train the models or
produce results reported in the associated manuscript — only the trained
model artifacts needed to run inference on new input data are provided.

## 12. Citation

If you use this software, please cite the associated publication. Citation
details will be added once the manuscript is published; the entry below is
a placeholder.

```bibtex
@article{cmsmark2026,
  title   = {<Paper Title>},
  author  = {<Authors>},
  journal = {<Journal Name>},
  year    = {2026}
}
```

## 13. License

This repository does not currently include a `LICENSE` file. Licensing
terms have not yet been specified; until a license is added, no license
should be assumed.

## 14. Authors / Contact

Repository history attributes this project to Dorsa Hayati
(dorsa.hayati@gmail.com). For questions specific to the associated
publication, please refer to the corresponding author information in that
manuscript once available.
