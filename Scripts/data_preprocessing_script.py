import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

def save_dataframe(x_train, y_train, x_test, y_test, features, name):
    X_train = pd.DataFrame(x_train)
    X_train.columns = features
    Y_train = pd.DataFrame(y_train)
    train_frame = [X_train, Y_train]
    train_final = pd.concat(train_frame,axis=1)
    train_final.to_csv(name +'_train.csv',index = False)
    print("Train dataset Saved")

    # Save test data set as well
    X_test = pd.DataFrame(x_test)
    X_test.columns= features
    Y_test = pd.DataFrame(y_test)
    test_frame = [X_test,Y_test]
    test_final = pd.concat(test_frame,axis=1)
    test_final.to_csv(name+'_test.csv',index = False)
    print("Test dataset Saved")

"""# Read Files and Basic Info"""
train_file_path = 'Train_ULAK.csv'
test_file_path = 'Test_ULAK.csv'
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

# number of each label type
train_df['Label'].value_counts()
test_df['Label'].value_counts()

"""# Data preprocessing
## Drop Records Containing Infinite Values, are Null and Nan Values
"""
# Replace infinite number with NaN values
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop null and NaN values
print("Corrupted records were deleted in training dataset")
print("Corrupted records were deleted in test dataset")
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

"""## Label-Encoding to Replace Categorical Values with Numerical"""
labels = train_df['Label'].values.reshape(-1, 1)
# Initialize the Label Encoder
encoder = LabelEncoder()
# transform fit labels of train and test daframe
train_labels = encoder.fit_transform(labels.reshape(-1,))
test_labels = encoder.transform(test_df['Label'].values.reshape(-1,))
label_order_classes = encoder.classes_
train_df['Label'] = train_labels
test_df['Label'] = test_labels
print(label_order_classes)
# Print the new sizes of train and test dataframes
print("Shape of Training Dataset:", train_df.shape)
print("Shape of Testing Dataset:", test_df.shape)

"""## Replace -1 values with the mean of their corresponding column"""
# Replace -1 with NaN
train_df.replace(-1, np.nan, inplace=True)
test_df.replace(-1, np.nan, inplace=True)
# Calculate the mean of each column
column_means = train_df.mean(skipna=True)
# Fill NaN values with the mean of the corresponding column
train_df.fillna(column_means, inplace=True)
test_df.fillna(column_means,inplace=True)

"""## Remove the features with zero values"""
columns_to_drop = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate',
                   'Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate']
# Drop the specified columns from the DataFrame
train_df.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=columns_to_drop,inplace=True)
# Print the new sizes of train and test dataframes
print("Shape of Training Dataset:", train_df.shape)
print("Shape of Testing Dataset:", test_df.shape)

"""## Remove duplicates"""
train_df = train_df.drop_duplicates().reset_index(drop=True)
print("Shape of Training Dataset:", train_df.shape)
train_df['Label'].value_counts()
train_df.head()

"""## Split Data"""
# number of each label type train df
train_df['Label'].value_counts()
X_train = train_df.drop(train_df.columns[-1], axis=1)
y_train = train_df[train_df.columns[-1]]
X_test = test_df.drop(test_df.columns[-1], axis=1)
y_test = test_df[test_df.columns[-1]]

"""## Save the first version of the dataset with the 70 features"""
save_dataframe(X_train,y_train,X_test,y_test,X_train.columns,'cleaned_dataset')

