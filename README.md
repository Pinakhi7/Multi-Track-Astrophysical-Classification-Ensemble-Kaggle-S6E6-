# Multi-Track Astrophysical Classification Ensemble (Kaggle S6E6)

An end-to-end Machine Learning production pipeline utilizing a **Dual-Track Level-0 Cross-Validated Ensemble** stacked via a **Level-1 Multinomial Logistic Blending Model**. The architecture is explicitly engineered to classify stellar objects into one of three deep-space target profiles: `GALAXY` (0), `QSO` (1), or `STAR` (2).

---

## Architectural Paradigm & Design Choices

To maximize model variance and capture distinct data structures without over-fitting, this pipeline isolates the data flow into two parallel paradigms within a strict **Stratified 5-Fold Cross-Validation** engine:

* **TRACK A: Tree-Based Gradient Boosting (Raw Feature Space)**
    * **Models:** LightGBM, XGBoost, and CatBoost.
    * **Logic:** Tree models excel at capturing complex non-linear feature interactions, step functions, and categorical thresholds directly from the raw engineered stellar features.
* **TRACK B: Deep Tabular Learning (Standardized Latent Space)**
    * **Model:** PyTorch Multi-Layer Perceptron (`RealMLP` with LayerNorm, Dropout, and ReLU).
    * **Logic:** Captures smooth, continuous linear transformations. Crucially, scaling transformations (`StandardScaler`) are calculated *strictly within each local training fold split* to prevent out-of-fold data leakage.

The out-of-fold (OOF) predicted probabilities from all 4 models are concatenated to form a 12-dimensional Level-1 meta-feature space, which is then combined using a highly regularized Multinomial Logistic Regression model.

---

## Repository Blueprint

```text
├── notebooks/
│   └── ensemble_pipeline.py     # Unified, GPU-accelerated production script
├── output/
│   └── submission.csv           # Final generated test inference predictions
├── README.md                    # Core project introduction & setup layout
└── REPORT.md                    # Deep-dive theoretical report & math analysis
Performance & Hardware Optimization
The script is fully optimized out-of-the-box for Kaggle Cloud Hardware Accelerators (NVIDIA Tesla T4 GPU backends):

Data Transfer Optimization: Eliminates row-by-row CPU-to-GPU data transfer warnings by stripping Pandas dataframe structures down to clean NumPy memory views (.values) right before feeding the XGBoost predictors.

GPU Binning: Leverages XGBoost's device='cuda' parameter and CatBoost's task_type='GPU' to shift tree-splitting tasks directly into graphics VRAM.

Throughput Saturation: The PyTorch DataLoader batch size is set to 512 to keep the GPU streaming multiprocessors consistently saturated, accelerating deep learning training cycles.

Quick Start Guide
1. Prerequisites & Environment Setup
Ensure you have the required compute frameworks installed in your Python runtime:

Bash
pip install numpy pandas torch scikit-learn lightgbm xgboost catboost
2. Data Organization
Download the data from the Kaggle competition page and place the source files into a local data/ directory:

Plaintext
data/
├── train.csv
└── test.csv
3. Pipeline Ingestion & Inference
Execute the training loop from your terminal console. The script will dynamically process all 5 folds, perform multi-track feature adjustments, compute out-of-fold log-losses, and export your leaderboard file:

Bash
python notebooks/ensemble_pipeline.py
Upon successful completion, check the output/ directory for your final compiled prediction file: submission.csv.
