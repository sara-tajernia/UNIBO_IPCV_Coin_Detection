# Euro Coin Detection, Classification, and Counting

This repository contains a classical computer-vision pipeline that detects euro coins in photographs, classifies their denominations, and computes the total value in each image. The final submission is [`assignment_module_v3.ipynb`](assignment_module_v3.ipynb).

The solution uses Hough circles, hand-crafted objectness, affine-normalized crops, colour/material features, RootSIFT, ORB, rotation-aware HOG, Cartesian and polar template matching, and a shared physical-size model. It does not use a neural network or pretrained vision model.

## Repository contents

| Path | Purpose |
| --- | --- |
| `assignment_module_v3.ipynb` | Final, fully executed notebook with the expanded ablation and failure audit |
| `assignment_module_v2.ipynb` | Previous version retained for comparison |
| `assignment_module.ipynb` | Original notebook version |
| `coin_dataset/reference_set/` | Eight labelled reference images, one per denomination |
| `coin_dataset/target_set/` | 142 target images |
| `coin_dataset/label.csv` | Manually created development labels used for calibration and evaluation |

## Setup

Python 3.10 or newer is recommended. Install the main dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib opencv-contrib-python jupyter
```

Open the final notebook and run all cells from top to bottom:

```bash
jupyter notebook assignment_module_v3.ipynb
```

Paths in the notebook are relative to the repository root, so start Jupyter from this directory. A complete fresh execution took about three minutes in the verification environment; runtime depends on CPU and OpenCV build.

## Pipeline

1. Generate normal and conditional fallback Hough-circle candidates.
2. Score candidates using colour/brightness contrast and boundary/internal edge evidence.
3. Remove near-duplicate candidates with non-maximum suppression.
4. Warp every reference and target coin to a common 256×256 normalized view.
5. Estimate the copper, Nordic-gold, or bimetallic material family.
6. Compare eligible denominations using physical size, hybrid templates, ORB, RootSIFT, and rotation-aware HOG.
7. Optimize one shared pixels-per-millimetre scale for all raw detections in an image.
8. Apply the scene- and family-specific final objectness threshold.
9. Report per-image denominations and monetary totals.

## Executed development-set results

The fresh v3 run produced:

- Exact images: **95 / 142 (66.90%)**
- Correct denomination instances: **277 / 328**
- Coin recall: **84.45%**
- Coin precision: **80.76%**
- Coin F1: **82.56%**
- Predicted total: **€146.49**
- Labelled total: **€133.64**

These are development-set measurements, not unbiased test results. The same manual labels were used to select scoring weights, biases, scale priors, and rejection thresholds. The prediction functions do not read `label.csv` at inference time, but the fitted constants retain information from it.

## Ablation study

Section 12 of v3 contains 23 atomic comparisons, six explicitly labelled grouped stress tests, a proposal-burden diagnostic, paired whole-image bootstrap intervals, exact McNemar tests with Holm correction, and a final-threshold sensitivity sweep. No ablated configuration is re-tuned.

The largest atomic F1 decreases were:

| Removed or simplified choice | F1 change |
| --- | ---: |
| Hard material-family gate | −20.27 points |
| Physical-size evidence | −15.20 points |
| HOG evidence | −15.20 points |
| Scene-size-specific calibration | −12.87 points |
| Affine normalization replaced by square resize | −11.33 points |
| ORB evidence | −11.33 points |
| HOG rotation search | −11.33 points |
| Shared image scale | −10.73 points |

The fitted zero-offset objectness threshold was best in the tested sweep: **82.56% F1 and 95 exact images**. Lower thresholds added false detections; higher thresholds increasingly reduced recall. The small Cartesian scale-perturbation ablation was the only reported row whose paired bootstrap interval reached zero; all other tested F1 decreases had intervals below zero on this development set.

## Complete failure audit

The notebook reports all 47 non-exact images in a full ledger with actual denominations, predictions, missing/extra multiset counts, raw and accepted proposal counts, value error, and cautious stage indicators.

The executed failure taxonomy was:

| Outcome | Images |
| --- | ---: |
| Exact | 95 |
| Denomination substitutions with equal counts | 34 |
| Over-count only | 7 |
| Under-count only | 2 |
| Mixed count and denomination errors | 4 |

The hardest denomination was `50cent` at **65.38% F1**. Mean absolute per-image value error was **€0.211**, with a mean signed bias of **+€0.097** per image.

The label CSV contains denomination multisets but no circle coordinates. Consequently, the notebook does not claim spatial detector precision, recall, IoU, or a true coin-to-coin confusion matrix. Raw/accepted count differences are diagnostics, not localization ground truth.

## Defensive-path verification

The final code cell executes 15 smoke tests covering the meaningful implemented guard paths:

- missing image input;
- blank scenes with no Hough detections;
- clipped-circle rejection;
- duplicate suppression;
- absent and insufficient descriptors;
- a forced OpenCV matcher exception;
- flat-template ZNCC;
- border-safe normalized crops;
- empty full-classifier and ablation-classifier paths.

All 15 tests passed in the saved v3 execution. These checks cover the pipeline's intended defensive branches; they do not promise support for arbitrary malformed arrays outside the documented image/detection contract.

## Reproducibility and interpretation

- Target images are sorted numerically.
- Random visual samples and OpenCV matching use fixed seeds.
- The ablation cache must reproduce all 142 baseline predictions exactly before comparisons run.
- Bootstrap resampling uses a fixed seed and keeps each image's coins together.
- The zero-offset threshold sweep is asserted to reproduce the full baseline.
- The notebook is saved with outputs from a clean, error-free top-to-bottom execution.

For a stronger generalization claim, reserve an unseen test set or use nested cross-validation, re-fit every reduced model on training folds, and report variation across folds.
