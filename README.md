# Hybrid Quantum-Classical Multi-Disease Detection with Variational Quantum Classifiers

A professional research-oriented machine learning project that explores how hybrid quantum-classical models can be used for medical disease prediction across multiple datasets, including diabetes, heart disease, kidney disease, and liver disease. The implementation combines classical machine learning baselines with Variational Quantum Classifiers (VQC) to study the practical potential of quantum machine learning for healthcare applications. [1]

## Overview

This project presents an end-to-end diagnostic pipeline for medical disease classification using both classical and quantum-enhanced learning methods. The workflow includes dataset preprocessing, missing-value handling, feature scaling, dimensionality reduction, classical model training, quantum circuit design, evaluation, model comparison, and deployment-oriented prediction examples. [1]

The core idea is to compare well-established classical models such as Logistic Regression and Random Forest with a hybrid quantum-classical VQC built using PennyLane. For the quantum pipeline, high-dimensional clinical features are reduced using PCA before being encoded into qubits, making the approach suitable for simulation and near-term NISQ-style experimentation. [1]

## Key Features

- Multi-disease prediction across diabetes, heart disease, kidney disease, and liver disease tasks. [1]
- Hybrid learning pipeline combining classical preprocessing with quantum classification. [1]
- Comparative evaluation of Logistic Regression, Random Forest, and VQC models. [1]
- PCA-based feature reduction for low-qubit quantum execution. [1]
- Performance reporting using accuracy, precision, recall, F1-score, confusion matrices, and ROC curves. [1]
- Deployment-ready saving of classical models, preprocessors, and example prediction workflow. [1]

## Workflow

### 1. Data Preprocessing

Each dataset is cleaned and transformed before training. The preprocessing pipeline includes handling missing values, encoding categorical variables, standardizing features, splitting data into training and testing sets, and applying PCA where required for the quantum model. [1]

### 2. Classical Models

Two baseline machine learning models are trained for comparison:

- Logistic Regression
- Random Forest Classifier

These models provide a strong reference point for evaluating whether the VQC adds value in practical healthcare prediction tasks. [1]

### 3. Quantum Model

The quantum component is implemented as a Variational Quantum Classifier using PennyLane. Classical input features are reduced to match the number of qubits, embedded into a quantum circuit using angle encoding, and passed through trainable entangling layers. The circuit output is then used for binary or multiclass-style medical prediction experiments, depending on the dataset setup. [1]

## Results

The experiments show that classical models generally outperform the current VQC configurations in overall predictive accuracy across the tested datasets. However, the VQC demonstrates that quantum-enhanced classification is feasible and can still produce competitive results on selected tasks, particularly as a proof of concept for hybrid medical AI workflows. [1]

### Reported Findings

- **Liver Disease**: Logistic Regression achieved about 0.735 accuracy with high recall, while Random Forest achieved about 0.744 accuracy with a more balanced precision-recall tradeoff. [1]
- **Kidney Disease**: Logistic Regression and Random Forest both achieved 1.000 accuracy on the test split, while the VQC achieved about 0.7625 accuracy with perfect precision and lower recall. [1]
- **VQC Behavior**: The quantum model often produced fewer false positives but missed more positive cases than the best classical models, indicating a more conservative prediction pattern in some experiments. [1]

These findings suggest that hybrid quantum models are promising for experimentation and research, but classical methods remain stronger baselines for current small-scale medical prediction tasks under the tested configurations. [1]

## Example Metrics

| Dataset | Model | Accuracy | Precision | Recall | F1-score |
|---------|-------|----------|-----------|--------|----------|
| Liver Disease | Logistic Regression | 0.7350 [1] | 0.7407 [1] | 0.9639 [1] | 0.8377 [1] |
| Liver Disease | Random Forest | 0.7436 [1] | 0.7732 [1] | 0.9036 [1] | 0.8333 [1] |
| Kidney Disease | Logistic Regression | 1.0000 [1] | 1.0000 [1] | 1.0000 [1] | 1.0000 [1] |
| Kidney Disease | Random Forest | 1.0000 [1] | 1.0000 [1] | 1.0000 [1] | 1.0000 [1] |
| Kidney Disease | VQC | 0.7625 [1] | 1.0000 [1] | 0.6200 [1] | 0.7654 [1] |

## Deployment Concept

The project includes a practical deployment-oriented workflow for classical models. The best-performing classical model, along with preprocessing components such as the imputer, scaler, and optional PCA transformer, is saved using `joblib` so that the full pipeline can be restored later for inference. [1]

A Flask-based API concept is also outlined for real-time prediction. In this setup, the application loads the saved model artifacts, preprocesses incoming patient data, and returns predictions through a web endpoint. The notebook also notes that VQC deployment is more complex because the quantum circuit must be reconstructed along with its trained parameters rather than directly serialized like a standard classical model. [1]

## Tech Stack

- Python
- scikit-learn
- PennyLane
- NumPy
- pandas
- Matplotlib / Seaborn
- joblib
- Flask

## Use Cases

- Medical disease prediction research
- Hybrid quantum-classical machine learning experiments
- Benchmarking VQC against classical models
- Exploring NISQ-era healthcare AI applications
- Educational demonstrations of quantum machine learning workflows

## Project Value

This repository is useful for researchers, students, and developers interested in the intersection of healthcare AI and quantum machine learning. It demonstrates how hybrid pipelines can be designed, evaluated, and interpreted using real medical datasets while maintaining a practical comparison against strong classical baselines. [1]

## Suggested README Tagline

**Hybrid Quantum-Classical Medical Prediction using Variational Quantum Classifiers, with comparative evaluation against classical machine learning baselines across multiple disease datasets.** [1]

# Example metrics table
| Dataset        | Model               | Accuracy | Precision | Recall | F1-score |
|---------------|---------------------|----------|-----------|--------|----------|
| Diabetes      | Logistic Regression | 0.6948   | 0.5778    | 0.4815 | 0.5253   |
| Diabetes      | Random Forest       | 0.7792   | 0.7083    | 0.6296 | 0.6667   |
| Diabetes      | VQC                 | (fill)   | (fill)    | (fill) | (fill)   |
| Heart Disease | Logistic Regression | (fill)   | (fill)    | (fill) | (fill)   |
| Heart Disease | Random Forest       | (fill)   | (fill)    | (fill) | (fill)   |
| Heart Disease | VQC                 | (fill)   | (fill)    | (fill) | (fill)   |
| Kidney Disease| Logistic Regression | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Kidney Disease| Random Forest       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Kidney Disease| VQC                 | 0.7625   | 1.0000    | 0.6200 | 0.7654   |
| Liver Disease | Logistic Regression | 0.7350   | 0.7407    | 0.9639 | 0.8377   |
| Liver Disease | Random Forest       | 0.7436   | 0.7732    | 0.9036 | 0.8333   |
| Liver Disease | VQC                 | (fill)   | (fill)    | (fill) | (fill)   |
