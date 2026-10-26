# BrainWorks: Quantifying the Infiltrative Margin in Paediatric Diffuse Glioma

[![Conference](https://img.shields.io/badge/MICCAI-2026-blue.svg)](https://conferences.miccai.org/2026/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official, training-free evaluation code for our paper presented at the **BrainWorks Workshop @ MICCAI 2026 (Strasbourg, France)**.

> **Paper Title:** From Edge to Gradient: Quantifying the Infiltrative Margin in Paediatric Diffuse Glioma
> **Authors:** Jesús Ortiz-Perona, Jorge Reig-Calatayud, Luis Álvarez-Mira, and José Francisco Cánovas-Cerezo

## Overview

Automated segmentation of paediatric high-grade gliomas (DMG/DIPG) is challenging because these tumours present indistinct, infiltrative margins. This repository provides the codebase to reproduce our signal-level analysis, proving that the paediatric tumour margin should be modelled as a graded, uncertain transition rather than a crisp, high-frequency edge.

The analysis consists of:
1. **Boundary Detectability:** Evaluating a Wavelet detail-energy detector against a generic Sobel gradient.
2. **Margin Conspicuity:** Stratifying detectability based on edge strength.
3. **Signed-Distance Transition Profiling:** Quantifying the multi-millimetre transition width of the infiltrative margin.
4. **Qualitative Visualization:** 2D mapping of edge detection responses.

## Installation & Requirements

The code is designed to run efficiently on a standard CPU. To run the experiments, install the required dependencies:

```bash
pip install numpy pandas scipy scikit-learn nibabel matplotlib PyWavelets
```

## Dataset Preparation

This code is built to run on the **BraTS-PEDs** dataset. Before running the notebook, ensure your data is organized as follows inside a `data/` directory:

```text
data/
│
├── BraTS-PED-00013-100/
│   ├── BraTS-PED-00013-100-t2f.nii.gz    # FLAIR volume
│   └── BraTS-PED-00013-100-seg.nii.gz    # Segmentation mask
│
├── BraTS-PED-00025-100/
│   ├── BraTS-PED-00025-100-t2f.nii.gz
│   └── BraTS-PED-00025-100-seg.nii.gz
...
```
*(Note: Only the T2-FLAIR and the segmentation masks are required for this specific analysis).*

## Usage

The core experiments are unified into a single, highly readable Jupyter Notebook. 

1. Clone this repository.
2. Place your BraTS-PEDs dataset in the `data/` folder (or update the `DATA_DIR` variable in the notebook).
3. Open `BrainWorks_Experiments.ipynb` (you can run it locally or via Google Colab).
4. Run all cells sequentially to reproduce the tables, metrics, and figures presented in the manuscript.

### Auxiliary Modules
The notebook relies on several custom Python scripts included in this repository:
* `mechanistic_controls.py`: AUC calculations for Wavelet vs. Sobel.
* `margin_stratification.py`: Conspicuity index ($\sigma$) extraction.
* `distance_profile.py`: Core logic for the signed-distance intensity profiling.
* `wavelet_vs_gradient_width.py`: Physical width estimation functions.

## Citation

If you find this code or our research helpful, please consider citing our work (official Springer proceedings will be available in December 2026):

```bibtex
@inproceedings{ortizperona2026brainworks,
  title={From Edge to Gradient: Quantifying the Infiltrative Margin in Paediatric Diffuse Glioma},
  author={Ortiz-Perona, Jes{\'u}s and Reig-Calatayud, Jorge and {\'A}lvarez-Mira, Luis and C{\'a}novas-Cerezo, Jos{\'e} Francisco},
  booktitle={MICCAI BrainWorks Workshop},
  year={2026},
  organization={Springer}
}
```
