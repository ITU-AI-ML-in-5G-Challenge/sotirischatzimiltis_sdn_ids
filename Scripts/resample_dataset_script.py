import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

def save_dataframe(x_train, y_train, x_test, y_test, features, name):
    # Save the final dataset with the 23 features
    X_train = pd.DataFrame(x_train)
    X_train.columns = features
    Y_train = pd.DataFrame(y_train)
    train_frame = [X_train, Y_train]
    train_final = pd.concat(train_frame,axis=1)
    train_final.to_csv(name +'train.csv',index = False)
    print("Train dataset Saved")

    # Save test data set as well
    X_test = pd.DataFrame(x_test)
    X_test.columns= features
    Y_test = pd.DataFrame(y_test)
    test_frame = [X_test,Y_test]
    test_final = pd.concat(test_frame,axis=1)
    test_final.to_csv(name+'test.csv',index = False)
    print("Test dataset Saved")

"""# Read Files and Basic Info"""

train_file_path = ''
test_file_path = ''

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

# number of each label type train df
train_df['label'].value_counts()

# number of each label type test df
test_df['label'].value_counts()

"""## Split Data"""

# number of each label type train df
train_df['label'].value_counts()

X_train = train_df.drop(train_df.columns[-1], axis=1)
y_train = train_df[train_df.columns[-1]]
X_test = test_df.drop(test_df.columns[-1], axis=1)
y_test = test_df[test_df.columns[-1]]


""" Under-sampling Approach """

"""1st approach Tomek's link """
print(sorted(Counter(y_train).items()))
tl = TomekLinks()
X_train_undertomek, y_train_undertomek = tl.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train_undertomek))
# Save
save_dataframe(X_train_undertomek, y_train_undertomek, X_test, y_test, X_train_undertomek.columns, 'undertomek')
del X_train_undertomek, y_train_undertomek

"""2nd approach Repeated Edited Nearest Neighbours"""
print(sorted(Counter(y_train).items()))
renn = RepeatedEditedNearestNeighbours(sampling_strategy=[0, 1])
X_train_underRENN, y_train_underRENN = renn.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_underRENN).items()))
# Save
save_dataframe(X_train_underRENN,y_train_underRENN,X_test,y_test,X_train_underRENN.columns,'underRENN')
del X_train_underRENN, y_train_underRENN

"""3rd approach Repeated Edited Nearest Neighbours"""
print(sorted(Counter(y_train).items()))
cnn = CondensedNearestNeighbour(sampling_strategy=[0, 1, 2])
X_train_underCNN, y_train_underCNN = cnn.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_underCNN).items()))
save_dataframe(X_train_underCNN, y_train_underCNN, X_test, y_test, X_train_underCNN.columns, 'underCNN')
del X_train_underCNN, y_train_underCNN

"""Over-sampling with SMOTE """
# adjust values of classes
smote = SMOTE(sampling_strategy={7: 5000, 8: 5000, 12: 5000, 13: 5000, 14: 5000}, random_state=42)
# #Apply SMOTE to the training data to create synthetic samples
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_resampled).items()))
save_dataframe(X_train_resampled, y_train_resampled, X_test, y_test, X_train_resampled.columns, 'SMOTE')
del X_train_resampled, y_train_resampled


"""## Combination of over and under Sampling"""
"""1st approach SMOKE and Tomek"""
smote_tomek = SMOTETomek(random_state=0)
print(sorted(Counter(y_train).items()))
X_train_smotetomek, y_train_smotetomek = smote_tomek.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_smotetomek).items()))
save_dataframe(X_train_smotetomek, y_train_smotetomek, X_test, y_test, X_train_smotetomek.columns, 'smotetomek')
del X_train_smotetomek, y_train_smotetomek


"""2nd approach SMOKE and Edited Nearest Neighbours"""
smote_enn = SMOTEENN(random_state=0)
print(sorted(Counter(y_train).items()))
X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_smoteenn).items()))
save_dataframe(X_train_smoteenn,y_train_smoteenn,X_test,y_test,X_train_smoteenn.columns,'smoteenn')
del X_train_smoteenn, y_train_smoteenn

