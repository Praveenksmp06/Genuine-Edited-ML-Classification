import pandas as pd

df = pd.read_csv("dataset_features/image_features.csv")

X = df.drop(["image_name", "label"], axis=1)
y = df["label"]

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_test_raw = X_test.copy()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print(y_pred_lr[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred_lr, zero_division=0)

recall = recall_score(y_test, y_pred_lr, zero_division=0)

f1 = f1_score(y_test, y_pred_lr, zero_division=0)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_prob_lr)
print("ROC-AUC Score:", auc)
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred_lr)

print("MCC Score:", mcc)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_dt = accuracy_score(y_test, y_pred_dt)

from sklearn.metrics import precision_score
precision_dt = precision_score(y_test, y_pred_dt, zero_division=0)

from sklearn.metrics import recall_score
recall_dt = recall_score(y_test, y_pred_dt, zero_division=0)

from sklearn.metrics import f1_score
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)

from sklearn.metrics import roc_auc_score
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
auc_dt = roc_auc_score(y_test, y_prob_dt)

from sklearn.metrics import matthews_corrcoef
mcc_dt = matthews_corrcoef(y_test, y_pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Precision:", precision_dt)
print("Decision Tree Recall:", recall_dt)
print("Decision Tree F1 Score:", f1_dt)
print("Decision Tree ROC-AUC Score:", auc_dt)
print("Decision Tree MCC Score:", mcc_dt)

from sklearn.neighbors import KNeighborsClassifier

kn_model = KNeighborsClassifier(n_neighbors=5)
kn_model.fit(X_train, y_train)
y_pred_kn = kn_model.predict(X_test)   
accuracy_kn = accuracy_score(y_test, y_pred_kn)
precision_kn = precision_score(y_test, y_pred_kn, zero_division=0)
recall_kn = recall_score(y_test, y_pred_kn, zero_division=0)
f1_kn = f1_score(y_test, y_pred_kn, zero_division=0)
y_prob_kn = kn_model.predict_proba(X_test)[:, 1]
auc_kn = roc_auc_score(y_test, y_prob_kn)
mcc_kn = matthews_corrcoef(y_test, y_pred_kn)

print("K-Nearest Neighbors Accuracy:", accuracy_kn)
print("K-Nearest Neighbors Precision:", precision_kn)
print("K-Nearest Neighbors Recall:", recall_kn)
print("K-Nearest Neighbors F1 Score:", f1_kn)
print("K-Nearest Neighbors ROC-AUC Score:", auc_kn)
print("K-Nearest Neighbors MCC Score:", mcc_kn) 

from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_pred_gnb = gnb_model.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
precision_gnb = precision_score(y_test, y_pred_gnb, zero_division=0)
recall_gnb = recall_score(y_test, y_pred_gnb, zero_division=0)
f1_gnb = f1_score(y_test, y_pred_gnb, zero_division=0)
y_prob_gnb = gnb_model.predict_proba(X_test)[:, 1]
auc_gnb = roc_auc_score(y_test, y_prob_gnb)
mcc_gnb = matthews_corrcoef(y_test, y_pred_gnb)

print("Gaussian NB Accuracy:", accuracy_gnb)
print("Gaussian NB Precision:", precision_gnb)
print("Gaussian NB Recall:", recall_gnb)
print("Gaussian NB F1 Score:", f1_gnb)
print("Gaussian NB ROC-AUC Score:", auc_gnb)
print("Gaussian NB MCC Score:", mcc_gnb)

from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0 ,probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, zero_division=0)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
auc_svm = roc_auc_score(y_test, y_prob_svm)
mcc_svm = matthews_corrcoef(y_test, y_pred_svm)

print("SVM Accuracy:", accuracy_svm)
print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1 Score:", f1_svm)
print("SVM ROC-AUC Score:", auc_svm)
print("SVM MCC Score:", mcc_svm)

import joblib
joblib.dump(lr_model, "models/logistic_regression_model.pkl")
joblib.dump(dt_model, "models/decision_tree_model.pkl")
joblib.dump(kn_model, "models/k_nearest_neighbors_model.pkl")
joblib.dump(gnb_model, "models/gaussian_nb_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

test_data = X_test_raw.reset_index(drop=True).copy()
test_data["label"] = y_test.reset_index(drop=True)
test_data.to_csv("test_data.csv", index=False)
print("Test data saved successfully!")
print("Test data shape:", test_data.shape)

results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc,
        "MCC": mcc
    },
    {
        "Model": "Decision Tree",
        "Accuracy": accuracy_dt,
        "Precision": precision_dt,
        "Recall": recall_dt,
        "F1-score": f1_dt,
        "ROC-AUC": auc_dt,
        "MCC": mcc_dt
    },
    {
        "Model": "KNN",
        "Accuracy": accuracy_kn,
        "Precision": precision_kn,
        "Recall": recall_kn,
        "F1-score": f1_kn,
        "ROC-AUC": auc_kn,
        "MCC": mcc_kn
    },
    {
        "Model": "Gaussian Naive Bayes",
        "Accuracy": accuracy_gnb,
        "Precision": precision_gnb,
        "Recall": recall_gnb,
        "F1-score": f1_gnb,
        "ROC-AUC": auc_gnb,
        "MCC": mcc_gnb
    },
    {
        "Model": "SVM",
        "Accuracy": accuracy_svm,
        "Precision": precision_svm,
        "Recall": recall_svm,
        "F1-score": f1_svm,
        "ROC-AUC": auc_svm,
        "MCC": mcc_svm
    }
]
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)
print("Model performance metrics saved successfully!")
print(results_df)
