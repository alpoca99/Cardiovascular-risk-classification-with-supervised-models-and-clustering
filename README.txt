# ==========================================
# CLINICAL PREDICTION AND CLUSTERING PROJECT
# ==========================================

## 1. Project Description
This project implements a comprehensive Machine Learning pipeline for data analysis, dimensionality reduction, clustering, and predictive classification. It is specifically designed to run in Jupyter Notebook environments. The workflow covers everything from data cleaning and scaling, through feature extraction (PCA, t-SNE, UMAP), to phenotype definition using K-Means and the training of ensemble algorithms (Random Forest, LightGBM, XGBoost) for outcome prediction, ensuring a rigorous metric evaluation scheme. KNN methodology utilizes NaN-Euclidean distance to calculate patient similarity vectors in incomplete multidimensional spaces.

IMPORTANT NOTE: supervised and reduction methodology.ipynb ---> first methodology and results with accuracy, ROC, f1-Score
                most recent run_validation_metrics.py ----> newest results with versions below and new metrics for validation (McNemar, sensitivity, Specificity)  

## 2. Environment Requirements
*   Language: Python 3.14
*   Execution Environment: Jupyter Notebook (.ipynb)
*   numpy >= 1.26
*   pandas  >= 2.0
*   scipy >= 1.11
*   scikit-learn >= 1.3
*   lightgbm >= 4.0
*   xgboost >= 2.0
*   umap-learn >= 0.5.5

# Dependencies and Libraries
To run this project, make sure to install the following libraries:
*   pandas
*   seaborn
*   numpy
*   umap-learn (imported as umap.umap_)
*   lightgbm
*   xgboost
*   scikit-learn
*   matplotlib
*   openpyxl

The exact imports used in the code are:

import pickle
import shutil
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import umap.umap_ as umap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## 3. File Structure and Execution Order

To ensure there are no data dependency errors (such as data leakage), the Jupyter Notebooks must be executed strictly in the following order:

### [0] 00_Data_processing_KNN.ipynb
*   What it does: Utilizes NaN-Euclidean distance to calculate patient similarity vectors in incomplete multidimensional spaces. Executes cross-validated parameter searching (_select_best_k) to dynamically select the optimal neighbor count $k$ for minimized reconstruction error.  Constrains imputed values for discrete nominal and binary attributes by rounding continuous matrix outputs back to valid discrete class boundaries.
*   Main libraries: pandas, scikit-learn, matplotlib, openpyxl
*   Pipeline transformation: Houses cleaning and imputation logic inside CVDDataPipeline to prevent data leakage during model training. Supports both label encoding and full one-hot encoding for downstream machine learning compatibility. Exports processed datasets along with serialized pickled pipeline objects for reproducible inference.
*   Validation: Artificially injects missingness into complete dataset rows by masking 15% of numerical cells and 10% of categorical cells. Measures quantitative imputation precision using Relative Root Mean Squared Error (RRMSE). Assesses categorical accuracy using exact-match classification accuracy scores.

### [1] 01_Preprocessing_and_Scaling.ipynb
*   What it does: Loads the original dataset, performs data cleaning, splits the dataset into training and testing sets using stratified sampling to prevent data leakage, and applies standardization.
*   Main libraries: pandas, numpy, train_test_split, StandardScaler.
*   Output: Clean, scaled, and split datasets (Train/Test) ready for modeling.

### [2] 02_Dimensionality_Reduction.ipynb
*   What it does: Takes the scaled training data and applies spatial dimensionality reduction techniques to find underlying patterns. It evaluates the reliability of the topological reduction.
*   Main libraries: PCA, umap.umap_, trustworthiness, matplotlib.pyplot.
*   Output: Spatial transformations and 2D/3D data visualizations.

### [3] 03_Phenotypic_Clustering.ipynb
*   What it does: Executes unsupervised learning algorithms on the reduced (or scaled) space to segment the population into distinct clinical clusters (phenotypes).
*   Main libraries: KMeans, seaborn, matplotlib.pyplot.
*   Output: Cluster labels assigned to patients and tables of demographic/clinical features per group.

### [4] 04_Classification_and_Validation.ipynb
*   What it does: Trains predictive Machine Learning models using the training set. Subsequently, it performs inference on the test set (incognito cohort) and evaluates the final performance.
*   Main libraries: RandomForestClassifier, LGBMClassifier, xgboost, classification_report, confusion_matrix, accuracy_score.
*   Output: Confusion matrices, final metrics report (Accuracy, Precision, Recall, F1-Score), and the optimized final model ready for production.

---
Execution Note: Make sure to restart the Jupyter kernel and run all cells in order (Run All) within each notebook so that global variables are instantiated correctly.
