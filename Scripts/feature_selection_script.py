import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
# use Stratified KFold instead to maintain proportions of each class in the folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def save_dataframe(x_train,y_train,x_test,y_test,features,name):
    # Save the final dataset with the 23 features
    X_train = pd.DataFrame(x_train)
    X_train.columns= features
    Y_train = pd.DataFrame(y_train)
    train_frame = [X_train,Y_train]
    train_final = pd.concat(train_frame,axis=1)
    train_final.to_csv('/content/drive/MyDrive/ITU_Competition_Intrusion_and_Vulnerability_Detection_in_Software_Defined_Networks(SDN)/train_data_' + name +'.csv',index = False)
    print("Train dataset Saved")

    # Save test data set as well
    X_test = pd.DataFrame(x_test)
    X_test.columns= features
    Y_test = pd.DataFrame(y_test)
    test_frame = [X_test,Y_test]
    test_final = pd.concat(test_frame,axis=1)
    test_final.to_csv('/content/drive/MyDrive/ITU_Competition_Intrusion_and_Vulnerability_Detection_in_Software_Defined_Networks(SDN)/test_data_'+ name+'.csv',index = False)
    print("Test dataset Saved")

"""# Read Files and Basic Info"""

train_file_path = '/content/drive/MyDrive/ITU_Competition_Intrusion_and_Vulnerability_Detection_in_Software_Defined_Networks(SDN)/cleaned_dataset_train.csv'
test_file_path = '/content/drive/MyDrive/ITU_Competition_Intrusion_and_Vulnerability_Detection_in_Software_Defined_Networks(SDN)/cleaned_dataset_test.csv'

train_df = pd.read_csv(train_file_path,skipinitialspace=True)
test_df = pd.read_csv(test_file_path,skipinitialspace=True)

if train_df is not None and test_df is not None:
    print("Data loaded successfully!")
    print("Train Data:")
    print(train_df.head())  # Display the first few rows of the train DataFrame
    print("\nTest Data:")
    print(test_df.head())   # Display the first few rows of the test DataFrame

# Print the sizes of train and test dataframes
print("Shape of Training Dataset:", train_df.shape)
print("Shape of Testing Dataset:", test_df.shape)

X_train = train_df.drop(train_df.columns[-1], axis=1)
y_train = train_df[train_df.columns[-1]]

X_test = test_df.drop(test_df.columns[-1], axis=1)
y_test = test_df[test_df.columns[-1]]

"""# Manual Recursive Feature Selection using Random Forest feature importance"""

# get the indexes of the five folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

"""Method for Random Forest Feature Importance
*   Uses 5-fold cross validation
*   Keeps features until cumulative importance >= 0.90


"""

def feature_selection(X,X_test,Y,Y_test,skf,name='default',criterion=0,cumulative=False):
    # Scale doesn't do anythign after the
    #first iteration since data already scaled
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(X_test)

    features = X.columns

    feature_importance_scores = np.zeros(X.shape[1])

    for train_index, val_index in skf.split(X,Y):
        X_train, X_val = x_train[train_index],x_train[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        # Train a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=20, max_depth=None,
                                 bootstrap=False, n_jobs=-1,
                                 random_state=0,verbose=2)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print("Accuracy:", accuracy)
        # Accumulate feature importance scores
        feature_importance_scores += rf.feature_importances_

    feature_importance_scores /= 5 # divide by 5 to find average F.E.S.

    # Create a DataFrame to store feature importance scores
    feature_importance_df = pd.DataFrame({'feature': features,
                'importance': feature_importance_scores})
    # Sort features by importance scores in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance',
                                    ascending=False)
    # Keep features based on either a criterion or based on culumative importance
    if cumulative:
        cumulative_importance = 0
        selected_features = []
        indexes = []
        for index, row in feature_importance_df.iterrows():
            if cumulative_importance >= 0.9:
                break
            indexes.append(index)
            selected_features.append(row['feature'])
            cumulative_importance += row['importance']
        return selected_features
    else:
        indexes = feature_importance_df['importance'] > criterion
        selected_features = feature_importance_df[indexes]['feature'].tolist()
        return selected_features
    # save data set if wanted:
    return

"""## First Iteration

"""

selected_features = feature_selection(X_train,X_test,y_train,y_test,skf,name='reduced',criterion=0,cumulative=True)

print("New features reduced to :",len(selected_features))
X_train = X_train.loc[:,selected_features]
X_test = X_test.loc[:,selected_features]
print(selected_features)
save_dataframe(X_train,y_train,X_test,y_test,selected_features,'reduced')
