## Helpful Documentation

### Requirements
- Sklearn (1.3.0)
- Numpy (1.24.3)
- Pandas (1.5.3)
- Seaborn (0.12.2)
- Matplotlib (3.7.1)
- xgboost (1.7.3)
- tabulate (0.8.10)
- psutil (5.9.0)
- imbens (0.11.0)

### Usage
#### Data Pre-processing
The following script removes instances with NaN or Infinte values, removes duplicate instances, replaces missing values (-1) with mean values and encodes categorical labels to numerical. 
1. Download the [data_preprocessing_script.py](https://github.com/sotirischatzimiltis/SDN_IDS/blob/main/Scripts/data_preprocessing_script.py).
2. Change paths to show to the correct train and test csv files.
    - train_file_path
    - test_file_path
3. Script will create two new csv files.
    -  cleaned_dataset_train.csv
    -  cleaned_datase_test.csv

#### Feature Selection
This script uses Random Forest feature importance to elliminate features that do not contribute in the overall training procedure.
1. Download the [feature_selection_script.py](https://github.com/sotirischatzimiltis/SDN_IDS/blob/main/Scripts/feature_selection_script.py).
2. Change paths accordingly.
    - train_file_path
    - test_file_path
3. Script will generate two new csv file.
    - train_data_reduced.csv
    - test_data_reduced.csv

#### Machine Learning 
This script trains 5 (five) different ML models. The models are evaluated using 5-fold stratified cross validation as well as an independent test set
The models used in the srcipt are: Decision Trees, Random Forest, K-Nearest Neighbours, Bagging Classifier and XGBoost.
1. Download the [ml_models_script.py](https://github.com/sotirischatzimiltis/SDN_IDS/blob/main/Scripts/ml_models_script.py).
2. Change paths accordingly for:
    - train_file_path
    - test_file_path
3. Script will create two text file for each model, one for cross-validation performance and one for test set evaluation. Furthermore the script will create two confusion matrices for each model.
   

     
