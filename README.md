# Cascaded 3D Deep Learning for Prostate mpMRI

A two-stage framework that segments the prostate and then classifies each case as benign or malignant. Developed on the public Prostate158 dataset by Olive Ibeto (pipeline design and implementation) in collaboration with Stanley Darlington.

> **Status: work in progress.** Preliminary results on a small public dataset. Research code, not for clinical use.

## Methods overview

- **Stage 1:** a residual 3D U-Net (MONAI) takes T2W, ADC and DWI volumes and segments the peripheral zone and central gland.
- **Stage 2:** the Stage 1 mask is used to crop the prostate from the T2W and ADC channels, and a 3D ResNet-10 outputs the probability of malignancy.
- **Experiments:** *cascade* (Stage 1 crops), *oracle* (crops from reference masks, i.e. a perfect Stage 1) and *baseline* (whole volume, no Stage 1).

## Data

Prostate158 (Adams et al., 2022): 139 training-archive cases, split 97 / 20 / 22 into training, validation and test sets (seed 42). The scan data are not included here; download them from the dataset's official page https://zenodo.org/records/6481141. Stage 2 labels are **annotation-based** (a case is malignant if a radiologist outlined a suspicious area, PI-RADS 4 or higher), not histopathology. The derived case-level labels are in `stage2_labels_check.csv`.

## Preliminary results

Stage 1: test whole-gland Dice 0.859 ± 0.035 (0.555 averaged over the two zones).

Stage 2 on 22 test cases:

| Experiment | AUC (95% CI) | Accuracy |
|---|---|---|
| Cascade | 0.620 (0.355 to 0.875) | 0.682 |
| Oracle | 0.620 (0.359 to 0.868) | 0.636 |
| Baseline | 0.479 (0.222 to 0.742) | 0.455 |

With only 22 test cases the confidence intervals include 0.5, so these results are not statistically conclusive. Full metrics are in `stage2_results.json`.

## Environment setup

The models were trained on a CPU, without a GPU.

```
pip install -r requirements.txt
```

## Running the code
   1. `main.py`: preprocessing (resampling, registration, normalization). Edit the input and output paths at the top.
   2. `train_stage1_unet.py`: Stage 1 training.
   3. `Cache.py`: saves the preprocessed cases to a `cache` folder for Stage 2.
   4. `train_stage2_classifier.py`: Stage 2 training and evaluation.

Edit the data path variables at the top of each script before running. Trained weights, cached files and scan data are not included.

## Limitations and next steps

- Small dataset, single data split, annotation-based labels.
- Planned: fine-tuning and external validation on a Nigerian tertiary-hospital mpMRI dataset, with pathology-based labels where available.


## References

- Adams, L. C., et al. (2022). Prostate158: an expert-annotated 3T MRI dataset and algorithm for prostate cancer detection. *Computers in Biology and Medicine*.
- Cardoso, M. J., et al. (2022). MONAI: an open-source framework for deep learning in healthcare. arXiv:2211.02701.