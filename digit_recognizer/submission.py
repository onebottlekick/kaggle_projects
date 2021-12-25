import argparse
import csv
import os
import pickle

import tensorflow as tf

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def submit(model_name, online, message):
    print('-'*20 + 'making submission' + '-'*20)
    with open('datasets/submission_data.pkl', 'rb') as f:
        data = pickle.load(f)
    data = data.reshape(-1, 28, 28, 1)
    data = data/255.0
    model_path = os.path.join('models', model_name)
    model = tf.keras.models.load_model(model_path + '.h5')
    
    predictions = model.predict(data).argmax(axis=-1)
    
    os.makedirs('datasets/submission', exist_ok=True)
    submit_file_path = os.path.join('datasets/submission/', model_name + '.csv')
    with open(submit_file_path, 'w', newline='') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, label in enumerate(tqdm(predictions, desc='making csv file')):
            writer.writerow({'ImageId':idx+1, 'Label':label})
            
    if online:
        os.system(f'kaggle competitions submit -c digit-recognizer -f {submit_file_path} -m "{message}"')
        print()
        os.system('kaggle competitions submissions digit-recognizer')
        print()

    print('Done!')  
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Kaggle auto submit')
    parser.add_argument('--model', type=str, help='model name', required=True)
    parser.add_argument('--online', type=bool, default=False, help='submit to kaggle.com')
    parser.add_argument('--message', type=str, default='submission', help='submission message')
    
    args = parser.parse_args()
    submit(args.model, args.online, args.message)
