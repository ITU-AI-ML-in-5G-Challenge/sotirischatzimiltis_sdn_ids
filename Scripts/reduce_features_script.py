import pandas as pd

# read test file (smaller) and get the feature names
file_path = '../Datasets/test_data_reduced.csv'
df = pd.read_csv(file_path,skipinitialspace=True)
features = df.columns.tolist()
del df

# read train and test file with all features you want to reduce
train_path = 'C:/Users/sc02449/OneDrive - University of Surrey/Desktop/ITU/smotetomektrain.csv'
test_path = 'C:/Users/sc02449/OneDrive - University of Surrey/Desktop/ITU/smotetomektest.csv'

train_df = pd.read_csv(train_path,skipinitialspace=True)
reduced_train_df = train_df[features]
del train_df
reduced_train_df.to_csv('reduced_resampled_data_train.csv', index=False)
del reduced_train_df

test_df = pd.read_csv(test_path,skipinitialspace=True)
reduced_test_df = test_df[features]
del test_df
reduced_test_df.to_csv('reduced_resampled_data_test.csv', index=False)
del reduced_test_df

