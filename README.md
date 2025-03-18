# Uterine Cancer Detection and Treatment Optimization

## Overview
This project focuses on developing a machine learning and reinforcement learning-based model to detect uterine cancer and optimize treatment recommendations. The model employs ensemble learning techniques for cancer detection and a Q-learning algorithm for treatment strategy optimization. SHAP (SHapley Additive exPlanations) is used to enhance model interpretability.

## Features
- **Synthetic Data Generation**: Uses `make_classification()` to create a dataset with 1000 samples and 20 features.
- **Feature Engineering**: Includes feature selection (RFE), dimensionality reduction (PCA), and patient subgrouping (K-Means clustering).
- **Handling Class Imbalance**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Machine Learning Models**:
  - Random Forest (RF)
  - XGBoost (Extreme Gradient Boosting)
  - LightGBM (Light Gradient Boosting Machine)
  - Support Vector Machine (SVM)
  - Stacking Ensemble for improved classification
- **Reinforcement Learning for Treatment Optimization**:
  - Implements a Q-learning agent to optimize treatment decisions
  - Defines a custom Gym environment for patient state transitions
- **SHAP Explainability**:
  - Uses SHAP values to analyze and visualize feature importance

## Installation
### Prerequisites
Ensure you have Python 3.7+ and the following libraries installed:
```bash
pip install numpy pandas matplotlib seaborn joblib shap gym imbalanced-learn xgboost lightgbm scikit-learn
```

## Usage
### 1. Train the Model
Run the script to train the machine learning models and save the best-performing model:
```bash
python train_model.py
```
This will generate:
- `optimized_uterine_cancer_model.pkl` (trained ensemble model)
- `scaler.pkl` (standardization model)

### 2. Evaluate Model Performance
The script evaluates accuracy, ROC AUC, classification reports, and confusion matrices.

### 3. Run Reinforcement Learning for Treatment Optimization
The Q-learning agent trains for 1000 episodes, optimizing treatment recommendations.

### 4. SHAP Analysis for Explainability
SHAP summary plots visualize feature importance in predictions.

## Outputs
- **Model Accuracy**: ~81.0%
- **ROC AUC Score**: ~0.89
- **Q-learning Convergence**: Demonstrates optimal policy learning for treatment decisions

## Future Improvements
- Implement Deep Q-Networks (DQN) for advanced reinforcement learning.
- Train on real-world clinical datasets for improved generalization.
- Enhance SHAP-based interpretability for clinical adoption.

## License
This project is open-source and available under the MIT License.

---
**Developed by:** Harshita Dubey

