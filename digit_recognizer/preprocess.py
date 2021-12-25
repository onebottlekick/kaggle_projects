import os
import pickle
import zipfile

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print('#'*80)
# download and extract data
os.makedirs('datasets', exist_ok=True)
os.system('kaggle competitions download -c digit-recognizer')
zip_file = 'digit-recognizer.zip'
extract_path = 'datasets'
with zipfile.ZipFile(zip_file, 'r') as zip:
    zip.extractall(extract_path)
os.remove(zip_file)

# data process
print('Processing Data...')
train_full = pd.read_csv('datasets/train.csv')
submission_data = pd.read_csv('datasets/test.csv')

X_full, y_full = train_full.drop('label', axis=1), train_full['label']
X_full, y_full = X_full.to_numpy(), y_full.to_numpy()
submission_data = submission_data.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# save processed data
print('Saving Data...')
dataset = [X_train, X_val, X_test, y_train, y_val, y_test]
with open('./datasets/data.pkl', 'wb') as f:
    pickle.dump(dataset, f)
    
with open('./datasets/submission_data.pkl', 'wb') as f:
    pickle.dump(submission_data, f)
print('#'*30 + '\tDone!\t' + '#'*30)