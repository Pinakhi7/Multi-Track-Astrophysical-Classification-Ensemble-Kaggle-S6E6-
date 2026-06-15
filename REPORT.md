#  Advanced Machine Learning Engineering Report
**Project Track:** Astrophysical Object Classification (Kaggle Playground Series S6E6)  
**Methodology:** Dual-Track Stratified Level-0 Ensembling with Level-1 Meta-Stacking  


## 1. Architectural Blueprint & System Data Flow

To ensure structural diversity and prevent cross-algorithm data contamination, the data processing flow is split into two completely isolated operational tracks inside a strict 5-fold stratified cross-validation loop. 

```text
                  [ Raw Data Ingestion: train.csv & test.csv ]
                                       |
                     [ Non-Linear Stellar Feature Engineering ]
                                       |
                       [ Unified One-Hot Column Alignment ]
                                       |
            ___________________________|___________________________
           |                                                       |
   [ TRACK A: Tree Space ]                                 [ TRACK B: Deep Learning Space ]
   - Sub-Models: LightGBM, XGBoost, CatBoost               - Sub-Model: PyTorch RealMLP
   - Transform: None (Raw Continuous Space)               - Transform: Strict Out-of-Fold Scaling
   - Optimization: GPU Bin Histograms                      - Optimization: High-Throughput Mini-Batching
           |                                                       |
           |___________________________ ___________________________|
                                       V
                  [ Stratified 5-Fold Cross-Validation Matrix ]
                                       |
                    [ Out-of-Fold Meta-Feature Extractions ]
                        - Meta-Train Shape: [N_Samples, 12]
                        - Meta-Test Shape:  [N_Test, 12]
                                       |
                  [ Level-1 Stacker: Multinomial Logistic Regression ]
                                       |
                             [ final_predictions.csv ]

```

---

## 2. Feature Engineering & Mathematical Foundations

Standard raw photometric magnitudes ($u, g, r, i, z$) describe localized flux but struggle to isolate physical attributes across shifting cosmic distances. The pipeline constructs domain-specific astronomical transformations to capture non-linear continuum behavior:

### I. Standard Color Indices

Calculated as sequential differences between neighboring bandpass filters:


$$\Delta m = \{u - g, \ g - r, \ r - i, \ i - z, \ u - r\}$$


These relative differences isolate color variations, serving as reliable proxies for effective stellar surface temperature, gravity, and ionization states while dampening distance-based variance.

### II. Total Integrated Brightness Proxy ($m_{\text{total}}$)

To track total optical energy output, raw magnitudes are converted back into linear flux space, summed, and transformed back into an integrated apparent magnitude using the logarithmic flux integration formula:


$$m_{\text{total}} = -2.5 \times \log_{10} \left( \sum_{b \in \{u,g,r,i,z\}} 10^{-0.4 \cdot b} \right)$$

### III. Redshift Interaction Mechanics

Formed by taking the element-wise product of color vectors against spectroscopic redshift ($z_{\text{spec}}$):


$$\text{Interaction} = \Delta m \times z_{\text{spec}}$$


This interaction captures cosmic velocity shifts, mapping how apparent colors alter across distance profiles. This provides strong separation boundaries between ultra-distant Quasars (`QSO`) and local stars (`STAR`).

### IV. Spectral Curvature

Calculated as second-order color spatial derivatives:


$$\text{Curv}_1 = (u - g) - (g - r)$$

$$\text{Curv}_2 = (g - r) - (r - i)$$


These metrics measure the acceleration of energy distribution across the optical spectrum, tracking subtle continuum variations unique to different deep-space sources.

---

## 3. Advanced Engineering Strategy: Dual-Track Processing

### Track A: Tree-Based Gradient Boosting (Raw Space)

* **Mechanics:** `LightGBM`, `XGBoost`, and `CatBoost` handle high-dimensional categories, non-linear boundaries, and skewed attributes natively without distance transformations.
* **Leakage Prevention:** Models ingest raw, un-transformed index slices inside the local cross-validation folds.

### Track B: Deep Tabular Neural Network (Standardized Space)

* **Mechanics:** Continuous features are fed to a custom PyTorch Multi-Layer Perceptron (`RealMLP`) containing sequential `Linear`, `LayerNorm`, `ReLU`, and `Dropout` layers.
* **Strict Anti-Leakage Protocol:** Neural networks require zero-mean, unit-variance inputs to prevent gradient saturation. However, scaling across the entire dataset introduces data leakage by exposing global distribution parameters (mean and standard deviation) to the local validation folds.
* **Implementation:** To enforce mathematical isolation, a new `StandardScaler` instance is instantiated inside every fold loop. It is fitted **exclusively** on the active training slice ($X\_{\text{tr\_trees}}$), and then applied to transform the validation slice ($X\_{\text{val\_trees}}$) and test matrix ($x\_{\text{test\_encoded}}$).

---

## 4. Hardware Optimization & Infrastructure Patches

Migrating from a local laptop compute environment to high-throughput cloud accelerators requires specific framework patches to prevent memory bottlenecks:

### I. The CUDA Capability Patch (`No Kernel Image Error`)

Older legacy cloud accelerators (such as the NVIDIA Tesla P100) run on Compute Capability `6.0`. Modern PyTorch Docker images are compiled with a minimum requirement of Compute Capability $\ge 7.0$. Shifting the cloud engine to **Tesla T4 GPUs** (Turing architecture, Compute Capability `7.5`) aligns the binary compilation, removing kernel mismatches and enabling stable tensor operations.

### II. Data Stream Optimization (`XGBoost Device Mismatch`)

Passing standard `Pandas DataFrames` directly to a GPU-mapped XGBoost configuration forces the environment to copy structural index metadata row-by-row from Host System RAM to Device VRAM. This stalls streaming processors.

> **The Fix:** Appending `.values` extracts raw, contiguous NumPy memory views. This feeds the arrays directly to CUDA VRAM blocks, bypassing dataframe tracking overhead entirely:

```python
model_xgb.fit(X_tr_trees.values, y_tr, eval_set=[(X_val_trees.values, y_val)], verbose=False)

```

### III. Execution Throughput Tuning

* **Batch Sizing:** Shifting the PyTorch `DataLoader` batch size from `30` to **`512`** ensures the T4 tensor cores stay saturated with data, reducing deep learning training times across folds.
* **Tree Histograms:** Utilizing `tree_method: 'hist'` inside XGBoost and setting `task_type: 'GPU'` in CatBoost maps feature bin-building algorithms onto GPU multi-threaded architectures.

---

## 5. Master Production Pipeline

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# =========================================================================
# 1. SETUP RAW DATA SPLITS 
# =========================================================================
train_df = pd.read_csv('/kaggle/input/competitions/playground-series-s6e6/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/playground-series-s6e6/test.csv')

x_raw = train_df.drop(columns=['id', 'class'])
y = train_df['class']
target_map = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
y_encoded = y.map(target_map)
x_test_raw = test_df.drop(columns=['id'])

# =========================================================================
# 2. FEATURE ENGINEERING ENGINE
# =========================================================================
def engineer_stellar_features(df):
    df_out = df.copy()
    
    df_out['u_g_index'] = df_out['u'] - df_out['g']
    df_out['g_r_index'] = df_out['g'] - df_out['r']
    df_out['r_i_index'] = df_out['r'] - df_out['i']
    df_out['i_z_index'] = df_out['i'] - df_out['z']
    df_out['u_r_index'] = df_out['u'] - df_out['r']

    df_out['total_brightness_proxy'] = -2.5 * np.log10(
        10**(-0.4 * df_out['u']) + 10**(-0.4 * df_out['g']) + 
        10**(-0.4 * df_out['r']) + 10**(-0.4 * df_out['i']) + 
        10**(-0.4 * df_out['z'])
    )

    df_out['u_g_redshift'] = df_out['u_g_index'] * df_out['redshift']
    df_out['g_r_redshift'] = df_out['g_r_index'] * df_out['redshift']
    df_out['spectral_curvature_1'] = df_out['u_g_index'] - df_out['g_r_index']
    df_out['spectral_curvature_2'] = df_out['g_r_index'] - df_out['r_i_index']
    
    return df_out

x_engineered = engineer_stellar_features(x_raw)
x_test_engineered = engineer_stellar_features(x_test_raw)

# =========================================================================
# 3. UNIFIED ONE-HOT ENCODING
# =========================================================================
combined = pd.concat([x_engineered, x_test_engineered], keys=['train', 'test'])
cat_col = ['spectral_type', 'galaxy_population']
combined_encoded = pd.get_dummies(combined, columns=cat_col, dtype=int)

x_encoded = combined_encoded.xs('train')
x_test_encoded = combined_encoded.xs('test')

num_cols = [
    'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 
    'u_g_index', 'g_r_index', 'r_i_index', 'i_z_index', 'u_r_index',
    'total_brightness_proxy', 'u_g_redshift', 'g_r_redshift',
    'spectral_curvature_1', 'spectral_curvature_2'
]

# =========================================================================
# 4. PYTORCH COMPONENTS FOR TRACK B
# =========================================================================
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        self.X = torch.tensor(X_val, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None: return self.X[idx], self.y[idx]
        return self.X[idx]

class RealMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, output_dim)
        
    def forward(self, x): return self.out(self.mlp(x))

def train_predict_nn_fold(X_train, y_train, X_val, X_test, epochs=30, batch_size=512, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test), batch_size=batch_size, shuffle=False)
    
    model = RealMLP(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_preds, test_preds = [], []
    with torch.no_grad():
        for batch_X in val_loader:
            val_preds.append(torch.softmax(model(batch_X.to(device)), dim=1).cpu().numpy())
        for batch_X in test_loader:
            test_preds.append(torch.softmax(model(batch_X.to(device)), dim=1).cpu().numpy())
            
    return np.vstack(val_preds), np.vstack(test_preds)

# =========================================================================
# 5. HARDWARE-OPTIMIZED HYPERPARAMETERS
# =========================================================================
lgb_params = {
    'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31,
    'max_depth': -1, 'feature_fraction': 0.8, 'random_state': 42, 'verbose': -1,
    'device': 'gpu'
}

xgb_params = {
    'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
    'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'random_state': 42, 
    'tree_method': 'hist', 'device': 'cuda'
}

cat_params = {
    'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
    'iterations': 500, 'learning_rate': 0.05, 'depth': 6,
    'random_seed': 42, 'verbose': False, 'task_type': 'GPU'
}

# =========================================================================
# 6. MASTER ENGINE CROSS-VALIDATION LOOP
# =========================================================================
N_SPLITS = 5
N_CLASSES = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(x_encoded), N_CLASSES))
oof_xgb = np.zeros((len(x_encoded), N_CLASSES))
oof_cat = np.zeros((len(x_encoded), N_CLASSES))
oof_nn  = np.zeros((len(x_encoded), N_CLASSES))

test_lgb = np.zeros((len(x_test_encoded), N_CLASSES))
test_xgb = np.zeros((len(x_test_encoded), N_CLASSES))
test_cat = np.zeros((len(x_test_encoded), N_CLASSES))
test_nn  = np.zeros((len(x_test_encoded), N_CLASSES))

print(" Running Accelerated Multi-Track Cross-Validation Pipelines...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(x_encoded, y_encoded)):
    print(f"--- Processing Fold {fold + 1} / {N_SPLITS} ---")
    
    # TRACK A: Tree Space Slices
    X_tr_trees, y_tr = x_encoded.iloc[train_idx], y_encoded.iloc[train_idx]
    X_val_trees, y_val = x_encoded.iloc[val_idx], y_encoded.iloc[val_idx]
    
    # LightGBM
    train_data_lgb = lgb.Dataset(X_tr_trees, label=y_tr)
    val_data_lgb = lgb.Dataset(X_val_trees, label=y_val, reference=train_data_lgb)
    model_lgb = lgb.train(lgb_params, train_data_lgb, num_boost_round=500, 
                          valid_sets=[val_data_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_val_trees)
    test_lgb += model_lgb.predict(x_test_encoded) / N_SPLITS
    
    # XGBoost (Optimized NumPy Array Pointers)
    model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=500, early_stopping_rounds=50)
    model_xgb.fit(X_tr_trees.values, y_tr, eval_set=[(X_val_trees.values, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_trees.values)
    test_xgb += model_xgb.predict_proba(x_test_encoded.values) / N_SPLITS
    
    # CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr_trees, y_tr, eval_set=(X_val_trees, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_trees)
    test_cat += model_cat.predict_proba(x_test_encoded) / N_SPLITS
    
    # TRACK B: Deep Learning Space Slices (Strict Anti-Leakage Scaling)
    scaler = StandardScaler()
    X_tr_nn = X_tr_trees.copy()
    X_val_nn = X_val_trees.copy()
    X_test_nn = x_test_encoded.copy()
    
    X_tr_nn[num_cols] = scaler.fit_transform(X_tr_trees[num_cols])
    X_val_nn[num_cols] = scaler.transform(X_val_trees[num_cols])
    X_test_nn[num_cols] = scaler.transform(x_test_encoded[num_cols])
    
    fold_nn_oof, fold_nn_test = train_predict_nn_fold(X_tr_nn, y_tr, X_val_nn, X_test_nn)
    oof_nn[val_idx] = fold_nn_oof
    test_nn += fold_nn_test / N_SPLITS
    
    print(f"Fold {fold + 1} processing complete.\n")

# =========================================================================
# 7. METABASE ASSEMBLY & STRUCTURAL STACKING
# =========================================================================
X_meta_train = np.hstack([oof_lgb, oof_xgb, oof_cat, oof_nn])
X_meta_test  = np.hstack([test_lgb, test_xgb, test_cat, test_nn])

print("\nFinal Layer: Training Regularized Meta-Model Stacker...")
meta_model = LogisticRegression(max_iter=1000, C=0.1, multi_class='multinomial', random_state=42)
meta_model.fit(X_meta_train, y_encoded)

meta_oof_probs = meta_model.predict_proba(X_meta_train)
print(f"Final Combined Meta-OOF Multi-LogLoss: {log_loss(y_encoded, meta_oof_probs):.5f}")

final_test_preds = meta_model.predict(X_meta_test)
inverse_target_map = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'class': pd.Series(final_test_preds).map(inverse_target_map)
})
submission_df.to_csv('submission.csv', index=False)
print(" Output matrix saved to 'submission.csv' for submission ingestion.")

```
