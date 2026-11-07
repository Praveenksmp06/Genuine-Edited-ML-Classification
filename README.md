# Genuine and Edited Image Classification Using Machine Learning

## 1. Project Overview

Digital images are widely used for communication, documentation, education, and sharing information. However, image editing tools make it increasingly easy to manipulate images without leaving obvious visual traces. Identifying whether an image is genuine or has been edited is therefore an important problem in digital image forensics.

This project focuses on classifying images into two categories: **Genuine** and **Edited**. It uses image-based numerical features and five supervised machine learning algorithms to identify possible image manipulations.

The project also includes a Streamlit web application through which users can upload a prepared feature dataset, select a trained machine learning model, view evaluation metrics, and explore prediction results.

### Project Objectives

- Develop a machine learning approach for genuine and edited image classification.
- Extract numerical features from images for classification.
- Train and evaluate five different classification algorithms.
- Compare model performance using multiple evaluation metrics.
- Identify the best-performing model based on the experimental results.
- Develop an interactive Streamlit application to present predictions and model evaluation results.

## 2. Problem Statement

Image manipulation can affect the reliability of digital images. Manually identifying edited images can be difficult, particularly when modifications are not visually obvious.

The objective of this project is to build a machine learning-based classification system that distinguishes genuine images from edited images using extracted image features.

The system evaluates different classification algorithms and compares their performance to understand which model provides the most suitable results for this binary classification problem.

## 3. Dataset Description

The project uses the CASIA 2.0 image forgery dataset, which contains genuine and tampered images.

The dataset used for this experiment contains:

| Category | Number of Images |
|---|---:|
| Genuine images (Au) | 7,491 |
| Edited images (Tp) | 5,123 |
| **Total** | **12,614** |

The images were processed to extract numerical features that could be used by traditional machine learning algorithms.

### Dataset Features

A total of **22 numerical features** were extracted from each image.

These features are grouped into the following categories:

| Feature Category | Description |
|---|---|
| Error Level Analysis (ELA) | Statistical information about image compression differences |
| Local Binary Pattern (LBP) | Texture-related information extracted from image regions |
| Grayscale Statistics | Statistical properties of grayscale image intensity |
| RGB Statistics | Mean intensity values of the red, green, and blue channels |

The extracted features are stored in a CSV file, which is used for model training, testing, and prediction.

## 4. Methodology

The project follows a sequence of image preprocessing, feature extraction, model training, evaluation, and prediction.

### 4.1 Image Feature Extraction

Numerical features are extracted from the images to represent their visual and statistical properties.

The extracted features include:

- ELA mean, standard deviation, minimum, maximum, median, and variance.
- Ten Local Binary Pattern (LBP) features.
- Grayscale mean, standard deviation, and entropy.
- Red, green, and blue channel mean values.

All 12,614 images were successfully processed, with no failed image extractions.

### 4.2 Data Preprocessing

After feature extraction, the generated CSV dataset was prepared for machine learning.

The following steps were performed:

1. Loaded the extracted image features into a DataFrame.
2. Separated the input features and target labels.
3. Used an 80:20 stratified train-test split.
4. Applied StandardScaler to standardize the numerical features.

The resulting dataset was divided as follows:

| Dataset | Number of Samples |
|---|---:|
| Training set | 10,091 |
| Testing set | 2,523 |
| Total | 12,614 |

The target variable contains two classes:

- `0` – Genuine
- `1` – Edited

### 4.3 Machine Learning Algorithms

Five supervised machine learning algorithms were implemented and evaluated.

**1. Logistic Regression**

Logistic Regression is a classification algorithm that estimates the probability of a sample belonging to a particular class. It is used as one of the baseline models for comparison.

**2. Decision Tree**

Decision Tree is a tree-based algorithm that makes classification decisions using a sequence of feature-based conditions.

**3. K-Nearest Neighbors (KNN)**

KNN classifies a sample based on the classes of its nearest neighboring samples in the feature space.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes is a probabilistic classification algorithm that applies Bayes' theorem while assuming a Gaussian distribution for the input features.

**5. Support Vector Machine (SVM)**

Support Vector Machine identifies a decision boundary that separates different classes. An RBF kernel was used for the SVM model.

## 5. Model Evaluation

The models were evaluated using six performance metrics:

- **Accuracy:** Measures the overall proportion of correctly classified samples.
- **Precision:** Measures how many samples predicted as positive actually belong to the positive class.
- **Recall:** Measures how many actual positive samples were correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across different classification thresholds.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classification while considering all four confusion matrix outcomes.

### 5.1 Model Comparison

The following table presents the test-set performance of all five classification algorithms.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8526 | 0.7837 | 0.8800 | 0.8290 | 0.9248 | 0.7038 |
| Decision Tree | 0.7911 | 0.7408 | 0.7473 | 0.7441 | 0.7842 | 0.5676 |
| KNN | 0.8022 | 0.7256 | 0.8254 | 0.7723 | 0.8764 | 0.6026 |
| Gaussian Naive Bayes | 0.7836 | 0.7236 | 0.7561 | 0.7395 | 0.8356 | 0.5550 |
| SVM | **0.8545** | 0.7760 | **0.9024** | **0.8345** | 0.9193 | **0.7124** |

### 5.2 Observations

The experimental results show differences in the performance of the five classification algorithms.

- SVM achieved the highest accuracy of approximately 85.45%.
- SVM also obtained the highest F1-score and MCC among the evaluated models.
- Logistic Regression achieved the highest ROC-AUC score of approximately 0.9248.
- Logistic Regression also obtained the highest precision.
- KNN achieved a moderate classification performance, with a recall of approximately 82.54%.
- Decision Tree and Gaussian Naive Bayes produced comparatively lower overall results in this experiment.

### 5.3 Best-Performing Model

Based on the evaluated test-set metrics, **Support Vector Machine (SVM)** was selected as the best-performing model.

It achieved:

- Accuracy: 85.45%
- Precision: 77.60%
- Recall: 90.24%
- F1-Score: 83.45%
- ROC-AUC: 91.93%
- MCC: 71.24%

Although Logistic Regression achieved a slightly higher ROC-AUC and precision, SVM provided the best overall balance across accuracy, recall, F1-score, and MCC in this experiment.

## 6. Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models and their performance.

The application allows users to upload a CSV file containing the extracted image features and select one of the five trained models.

### Application Features

**1. CSV Dataset Upload**

Users can upload a CSV file containing the numerical image features required for classification.

**2. Model Selection**

The application provides a selection option for the five trained models:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

**3. Performance Metrics**

After selecting a model, the application displays its evaluation metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and MCC.

**4. Prediction Summary**

The application presents the classification results, showing the number of samples predicted as genuine and edited.

**5. Confusion Matrix**

A confusion matrix is displayed to help understand correct and incorrect classifications.

**6. Classification Report**

The application provides a classification report containing precision, recall, F1-score, and support for each class.

**7. Model Comparison**

A comparison table and accuracy chart are provided to examine the performance of all five trained models.

### Live Application

**Live URL:** https://genuine-edited-ml-classification.streamlit.app/

## 7. Project Structure

```text
Genuine-Edited-ML-Classification/
│
├── app.py
├── count_images.py
├── extract_features.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── k_nearest_neighbors_model.pkl
│   ├── gaussian_nb_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
│
├── results/
│   └── model_comparison.csv
│
└── dataset_features/
    └── image_features.csv
```

## 8. Conclusion

This project demonstrates the application of traditional machine learning algorithms to genuine and edited image classification.

A total of 22 numerical features were extracted from 12,614 images and used to train and evaluate five classification models.

Among the evaluated algorithms, Support Vector Machine achieved the highest accuracy, F1-score, and Matthews Correlation Coefficient. Logistic Regression achieved the highest ROC-AUC and precision.

The Streamlit application provides an interactive interface for uploading feature datasets, selecting trained models, viewing predictions, and comparing model performance.

The results demonstrate how image-derived statistical and texture features can be used with machine learning techniques for image forgery classification. The system is an experimental classification approach and should not be considered a guarantee of detecting every form of image manipulation.

## 9. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Image Processing:** Pillow, OpenCV, scikit-image
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Development Environment:** VS Code, Jupyter Notebook
- **Version Control:** Git and GitHub

## 10. Repository Information

**Project:** Genuine and Edited Image Classification Using Machine Learning

**GitHub Repository:**  
https://github.com/Praveenksmp06/Genuine-Edited-ML-Classification

**Deployment:** https://genuine-edited-ml-classification.streamlit.app/

---
