# Import Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import time
import psutil
from imbens.metrics import classification_report_imbalanced
from sklearn import preprocessing
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

# Read Dataset
train_file_path = '../Datasets/train_data_reduced.csv'
test_file_path = '../Datasets/test_data_reduced.csv'
train_df = pd.read_csv(train_file_path, skipinitialspace=True)
test_df = pd.read_csv(test_file_path, skipinitialspace=True)

# Split data into X(input features) and Y (labels)
X_train = train_df.drop(train_df.columns[-1], axis=1)
y_train = train_df[train_df.columns[-1]]
X_test = test_df.drop(test_df.columns[-1], axis=1)
y_test = test_df[test_df.columns[-1]]
print(sorted(Counter(y_train).items()))
# Scale data
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

features = X_train.columns.tolist()
class_labels = ['Benign', 'Bot', 'DDoS', 'DoS_Golden_Eye', 'DoS_Hulk', 'DoS_Slowhttptest', 'DoS_Slowloris',
                'FTP_Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH_Patator', 'WebAttack_Brute_Force',
                'WebAttack_Sql_Injection', 'WebAttack_XSS']


# Performance evaluation method
def perf_evaluation(y_true, y_pred, class_labels, name, plot=False):
    # # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred, normalize='true')
    conf_matrix = confusion_matrix(y_true, y_pred)
    if plot:
        print("Confusion Matrix:")
        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(15, 15))
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap="viridis", square=True,
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        path = str(name) + '_cf.png'
        plt.savefig(path)
        plt.close()
    # Accuracy and Classification Report
    accuracy = accuracy_score(y_true, y_pred) * 100
    print("Total Accuracy: ", accuracy)
    report = classification_report(y_true, y_pred)
    # Classification Report based on the imbens library
    report_imb = classification_report_imbalanced(y_true, y_pred)
    if plot:
        print("Classification Report")
        print(report)
        print("Classification Report based on imbens")
        print(report_imb)
        lines = report_imb.strip().split('\n')  # Split string by lines
        headers = lines[0].split()  # extracts the headers
        data_list = []
        for line in lines[2:-2]:
            values = line.split()  # Split the line by spaces
            metrics = [float(value) for value in values[1:]]  # Convert numeric values to float
            data_list.append(metrics)  # Add metrics to the data array
        data_array = np.array(data_list)
        table = tabulate(data_array, headers=headers,tablefmt='plain', floatfmt=".3f")
        output_file_path = str(name) + '.txt'
        with open(output_file_path, "w") as file:
            file.write(table)
        print("Table written to", output_file_path)
        print(table)
        # Calculate TP, TN, FP, FN for each class
        class_tp = {}
        class_tn = {}
        class_fp = {}
        class_fn = {}
        for i, label in enumerate(class_labels):
            tp = conf_matrix[i, i]
            tn = conf_matrix.sum() - conf_matrix[i, :].sum() - conf_matrix[:, i].sum() + tp
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            class_tp[label] = tp
            class_tn[label] = tn
            class_fp[label] = fp
            class_fn[label] = fn
        # Print true positives, true negatives, false positives, and false negatives for each class
        for label in class_labels:
            print(f"Class {label}:")
            print("True Positives (TP):", class_tp[label])
            print("True Negatives (TN):", class_tn[label])
            print("False Positives (FP):", class_fp[label])
            print("False Negatives (FN):", class_fn[label])
            print()
    return matrix, report_imb, accuracy


# Performance evaluation auxiliary method to combine the folds
def perf_evaluation_auxiliary(matrices, reports, class_labels, name):
    stack_matrices = np.stack(matrices)
    average_matrix = np.mean(stack_matrices, axis=0)
    plt.figure(figsize=(15, 15))
    sns.heatmap(average_matrix, annot=True, fmt='.4f', cmap="viridis", square=True,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    path = str(name) + '_cf.png'
    plt.savefig(path)
    plt.close()

    # Initialize a dictionary to store the sum of metrics for each label
    sum_metrics = []
    headers = []
    # Count the number of dictionaries
    num_reports = len(reports)
    # Loop through each string report in the list
    for string_report in reports:
        # Since each report is a giant string first we need to bring it to the correct format
        lines = string_report.strip().split('\n')  # Split string by lines
        headers = lines[0].split()  # extracts the headers
        data_list = []
        for line in lines[2:-2]:
            values = line.split()  # Split the line by spaces
            metrics = [float(value) for value in values[1:]]  # Convert numeric values to float
            data_list.append(metrics)  # Add metrics to the data array
        data_array = np.array(data_list)
        sum_metrics.append(data_array)
    array_stack = np.stack(sum_metrics)
    average_report = np.mean(sum_metrics, axis=0)
    table = tabulate(average_report, headers=headers, tablefmt='plain', floatfmt=".3f")
    output_file_path = str(name) + '.txt'
    with open(output_file_path, "w") as file:
        file.write(table)
    print("Table written to", output_file_path)
    print(table)


# Cross Validation Method
def cross_validation(clf, X_train, y_train, skf, class_labels,name):
    matrices = []
    reports = []
    avg_accuracy = 0
    for train_index, val_index in skf.split(x_train, y_train):
        X_train_seg, X_val = x_train[train_index], x_train[val_index]
        y_train_seg, y_val = y_train[train_index], y_train[val_index]
        # Create base classifier
        c_clf = clf
        c_clf.fit(X_train_seg, y_train_seg)  # Fit data
        y_val_pred = c_clf.predict(X_val)  # Predict labels for validation data
        m, r, acc = perf_evaluation(y_val, y_val_pred, class_labels, name, plot=False)  # Print performance metrics
        matrices.append(m)
        reports.append(r)
        avg_accuracy += acc
    avg_accuracy = avg_accuracy / 5
    print("Average Accuracy 5-Fold Cross Validation: ", avg_accuracy)
    perf_evaluation_auxiliary(matrices, reports, class_labels, name)


# Test Evaluation Method
def test_evaluation(clf, X_train, y_train, X_test, y_test, class_labels,name):
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print("Training Time:", training_time, "seconds")
    start_time = time.time()
    y_pred = clf.predict(X_test)  # change to x_test_0
    end_time = time.time()
    prediction_time = end_time - start_time
    print("Prediction Time:", prediction_time, "seconds")
    perf_evaluation(y_test, y_pred, class_labels, name, plot=True)


# Main Code
# Initialize Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

# Cross Validation
print('Decision Tree Cross Validation')
DT_clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
cross_validation(DT_clf, X_train, y_train, skf, class_labels, 'dt_cv_reduced')

print('Random Forest Cross Validation')
RF_clf = RandomForestClassifier(class_weight='balanced_subsample', max_depth=None, n_estimators=50, random_state=0,
                                verbose=2, n_jobs=-1)
cross_validation(RF_clf, X_train, y_train, skf, class_labels, 'rf_cv_reduced')

print('K-NN Cross Validation')
KNN_clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1, weights='distance', n_jobs=-1)
cross_validation(KNN_clf, X_train, y_train, skf, class_labels, 'knn_cv_reduced')

print('Bagging Classifier Cross Evaluation')
BG_clf = BaggingClassifier(n_jobs=-1,verbose=2)
cross_validation(BG_clf, X_train, y_train, skf, class_labels, 'bagg_cv_reduced')

print('XGBoost Cross Validation Evaluation')
XGB_clf = XGBClassifier()
cross_validation(XGB_clf, X_train, y_train, skf, class_labels, 'xgb_cv_reduced')

# Test Set Evaluation
print('Decision Tree Test Evaluation')
DT_clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
test_evaluation(DT_clf, X_train, y_train, X_test, y_test, class_labels, 'dt_test_reduced')

print('Random Forest Test Evaluation')
RF_clf = RandomForestClassifier(class_weight='balanced_subsample', max_depth=None, n_estimators=50, random_state=0,
                                verbose=2, n_jobs=-1)
test_evaluation(RF_clf, X_train, y_train, X_test, y_test, class_labels, 'rf_test_reduced')

print('K-NN Test Evaluation')
KNN_clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1, weights='distance', n_jobs=-1)
test_evaluation(KNN_clf, X_train, y_train, X_test, y_test, class_labels, 'knn_test_reduced')

print('Bagging Classifier Test Evaluation')
BG_clf = BaggingClassifier(n_jobs=-1, verbose=2)
test_evaluation(BG_clf, X_train, y_train, X_test, y_test, class_labels, 'bagg_test_reduced')

print('XGBoost Test Evaluation')
XGB_clf = XGBClassifier()
test_evaluation(XGB_clf, X_train, y_train, X_test, y_test, class_labels, 'xgb_test_reduced')
