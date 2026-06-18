# 🧠 SSVEP-PSO-SVM-LOSO

## Subject-Independent SSVEP Classification Using LOSO Cross-Validation and PSO-SVM

This repository presents a MATLAB-based framework for subject-independent Steady-State Visual Evoked Potential (SSVEP) classification using Leave-One-Subject-Out (LOSO) cross-validation and Particle Swarm Optimization-based Support Vector Machine (PSO-SVM). The framework integrates signal preprocessing, feature extraction, dimensionality reduction, hyperparameter optimization, statistical validation, and explainable AI analysis.

---

##  Overview

Brain-Computer Interface (BCI) systems based on SSVEP enable users to communicate and interact with external devices using EEG signals. However, subject variability remains a major challenge in developing generalized classification models.

This work proposes a subject-independent classification framework based on:

* LOSO Cross-Validation
* PSO-SVM Classification
* Time and Frequency Domain Feature Extraction
* Principal Component Analysis (PCA)
* Statistical Validation
* Explainable AI (XAI) Analysis

---

##  Methodology

```text
Raw EEG Signals
        │
        ▼
Bandpass Filtering
        │
        ▼
Wavelet Denoising
        │
        ▼
Window Segmentation
        │
        ▼
Feature Extraction
(Time + FFT + PSD)
        │
        ▼
Z-Score Normalization
        │
        ▼
PCA
        │
        ▼
LOSO Cross Validation
        │
        ▼
PSO-SVM Classification
        │
        ▼
Performance Evaluation
        │
        ▼
Statistical Validation
        │
        ▼
Explainable AI Analysis
```

---

##  Repository Structure

```text
SSVEP-PSO-SVM-LOSO/

├── README.md

├── requirements.md

├── src/
│   ├── pso-svm-main.m
│   ├── fourpreprocss.m
│   ├── shapfinal.m
│   └── statictest.m

├── results/
│   ├── figures/
│   └── tables/

├── docs/

└── data/
```

---

##  Feature Extraction

The following EEG features are extracted:

### Time-Domain Features

* Mean
* Variance
* Skewness
* Kurtosis
* RMS
* Energy

### Frequency-Domain Features

* FFT Features
* Power Spectral Density (PSD)

### Dimensionality Reduction

* Principal Component Analysis (PCA)

---

##  Classification Framework

### Proposed Method

* Particle Swarm Optimization Support Vector Machine (PSO-SVM)

### Baseline Models

* Linear SVM
* K-Nearest Neighbors (KNN)
* Random Forest (RF)

### Validation Strategy

* Leave-One-Subject-Out (LOSO) Cross-Validation

---

## Statistical Validation

The following statistical analyses are performed:

| Metric                    | Purpose                          |
| ------------------------- | -------------------------------- |
| Mean ± Standard Deviation | Performance variability          |
| 95% Confidence Interval   | Uncertainty quantification       |
| Cohen’s Kappa             | Agreement beyond chance          |
| Kruskal–Wallis Test       | Class-wise significance analysis |
| Friedman Test             | Stability analysis               |
| McNemar Test              | Pairwise classifier comparison   |
| Bonferroni Correction     | Multiple comparison correction   |
| Post-hoc Power Analysis   | Sample size adequacy             |

---

##  Explainable AI (XAI)

A SHAP-inspired permutation importance framework is used to identify the most influential EEG features contributing to SSVEP classification.

### Generated Visualizations

* Global Feature Importance
* SHAP-like Beeswarm Plot
* Feature Contribution Analysis
* Feature Ranking

---

## 📈 Example Results

| Metric             | Value  |
| ------------------ | ------ |
| Mean Accuracy      | 98.30% |
| Standard Deviation | 4.04%  |
| Cohen’s Kappa      | 0.9773 |
| Macro Precision    | 0.9840 |
| Macro Recall       | 0.9830 |
| Macro F1-Score     | 0.9831 |

---

## 💻 Software Requirements

* MATLAB R2022b or later

### Required Toolboxes

* Signal Processing Toolbox
* Statistics and Machine Learning Toolbox
* Wavelet Toolbox

### Optional Toolboxes

* Parallel Computing Toolbox
* Bioinformatics Toolbox

---

##  Running the Code

Open MATLAB and execute:

```matlab
main
```

The framework automatically performs:

1. EEG preprocessing
2. Feature extraction
3. PCA
4. LOSO validation
5. PSO-SVM optimization
6. Statistical validation
7. Explainability analysis
8. Result visualization

---
Dataset link:

https://data.mendeley.com/datasets/px9dpkssy8/draft?a=7140665d-a0f0-40b2-a9fd-a731d21b6222

##  Citation

If you use this repository in your research, please cite:

```bibtex
@software{Ravichander2026,
  author = {Ravichander Janapati},
  title = {SSVEP-PSO-SVM-LOSO: Subject-Independent SSVEP Classification Framework},
  year = {2026},
  url = {https://github.com/yourusername/SSVEP-PSO-SVM-LOSO}
}
```

---

## 👨‍🔬 Author

**Dr. Ravichander J**

Department of Computer Science and Engineering

SR University, Warangal, India

Research Areas:

* Brain-Computer Interfaces (BCI)
* EEG Signal Processing
* Machine Learning
* Explainable AI
* Cognitive Computing

---

## 📜 License

This project is released for academic and research purposes.

Feel free to use, modify, and extend the code with proper citation.
