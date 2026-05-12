```markdown
# XAI-Based Glaucoma Diagnosis via Multi-Input DNN with SERS Spectra

This repository contains the official implementation of the multi-input deep learning framework for glaucoma diagnosis using surface-enhanced Raman scattering (SERS) spectra of aqueous humor (AH). This project integrates Explainable AI (XAI) to optimize model architecture and ensure biochemical interpretability.

## 📌 Project Overview
Glaucoma diagnosis traditionally relies on structural imaging, which often captures irreversible damage. Our approach utilizes the biochemical signatures within the intraocular microenvironment (AH) to establish a label-free, highly accurate (97.1% accuracy, AUC 0.986) diagnostic platform.

### Key Features
- **XAI-Driven Feature Extraction**: Utilizes first-layer weight analysis to identify core spectral regions (e.g., glutamine and acetylcarnitine-related peaks).
- **Multi-Input DNN Architecture**: Parallelly processes the whole spectrum and specific core feature regions to maximize diagnostic performance.
- **Biochemical Validation**: Mathematical evidence is cross-referenced with standard Raman peaks of glaucoma-related metabolites.

## 🛠 Installation
```bash
git clone [https://github.com/your-username/XAI-Glaucoma-SERS.git](https://github.com/your-username/XAI-Glaucoma-SERS.git)
cd XAI-Glaucoma-SERS
pip install -r requirements.txt

```

## 🚀 Usage

### 1. Data Preprocessing

Prepare your SERS spectral data in `.csv` or `.npy` format.

```bash
python preprocess.py --input ./raw_data --output ./processed_data

```

### 2. Model Training & XAI Analysis

To train the multi-input DNN and perform first-layer weight analysis:

```bash
python train.py --model multi_input --epochs 100 --batch_size 32

```

## 📊 Results

Our model achieves exceptional performance across independent datasets:

* **Accuracy**: 97.1%
* **Sensitivity**: 97.9%
* **AUC**: 0.986

The XAI analysis identified critical diagnostic regions at **399.25–600.50 cm⁻¹** and **1,998.75–2,401.25 cm⁻¹**.

