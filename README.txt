# ==========================================
# CLINICAL PREDICTION AND CLUSTERING PROJECT
# ==========================================

## 1. Project Description
This project implements a comprehensive Machine Learning pipeline for data analysis, dimensionality reduction, clustering, and predictive classification. It is specifically designed to run in Jupyter Notebook environments. The workflow covers everything from data cleaning and scaling, through feature extraction (PCA, t-SNE, UMAP), to phenotype definition using K-Means and the training of ensemble algorithms (Random Forest, LightGBM, XGBoost) for outcome prediction, ensuring a rigorous metric evaluation scheme.

## 2. Environment Requirements
*   **Language:** Python 3.14
*   **Execution Environment:** Jupyter Notebook (`.ipynb`)

### Dependencies and Libraries
To run this project, make sure to install the following libraries:
*   pandas
*   seaborn
*   numpy
*   umap-learn (imported as umap.umap_)
*   lightgbm
*   xgboost
*   scikit-learn
*   matplotlib

The exact imports used in the code are:
import pandas as pd
import seaborn as sns
import numpy as np
import umap.umap_ as umap
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## 3. File Structure and Execution Order

To ensure there are no data dependency errors (such as data leakage), the Jupyter Notebooks must be executed strictly in the following order:

### [1] 01_Preprocessing_and_Scaling.ipynb
*   **What it does:** Loads the original dataset, performs data cleaning, splits the dataset into training and testing sets using stratified sampling to prevent data leakage, and applies standardization.
*   **Main libraries:** `pandas`, `numpy`, `train_test_split`, `StandardScaler`.
*   **Output:** Clean, scaled, and split datasets (Train/Test) ready for modeling.

### [2] 02_Dimensionality_Reduction.ipynb
*   **What it does:** Takes the scaled training data and applies spatial dimensionality reduction techniques to find underlying patterns. It evaluates the reliability of the topological reduction.
*   **Main libraries:** `PCA`, `TSNE`, `umap.umap_`, `trustworthiness`, `matplotlib.pyplot`.
*   **Output:** Spatial transformations and 2D/3D data visualizations.

### [3] 03_Phenotypic_Clustering.ipynb
*   **What it does:** Executes unsupervised learning algorithms on the reduced (or scaled) space to segment the population into distinct clinical clusters (phenotypes).
*   **Main libraries:** `KMeans`, `seaborn`, `matplotlib.pyplot`.
*   **Output:** Cluster labels assigned to patients and tables of demographic/clinical features per group.

### [4] 04_Classification_and_Validation.ipynb
*   **What it does:** Trains predictive Machine Learning models using the training set. Subsequently, it performs inference on the test set (incognito cohort) and evaluates the final performance.
*   **Main libraries:** `RandomForestClassifier`, `LGBMClassifier`, `xgboost`, `classification_report`, `confusion_matrix`, `accuracy_score`.
*   **Output:** Confusion matrices, final metrics report (Accuracy, Precision, Recall, F1-Score), and the optimized final model ready for production.

---
**Execution Note:** Make sure to restart the Jupyter kernel and run all cells in order (Run All) within each notebook so that global variables are instantiated correctly.