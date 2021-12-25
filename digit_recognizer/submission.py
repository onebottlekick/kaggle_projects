import argparse
import csv
import os
import pickle
import textwrap

import tensorflow as tf

def submit(model_name, online, path, verbose):
    with open('datasets/submission_data', 'rb') as f:
        data = pickle.load(f)
    data = data.reshape(-1, 28, 28, 1)
    data = data/255.0
    model_path = os.path.join('models', model_name)
    model = tf.keras.models.load_model(model_path + '.h5')
    
    predictions = model.predict(data).argmax(axis=-1)
    
    submit_file_path = os.path.join(path, model_name + '.csv')
    with open(submit_file_path, 'w', newline='') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, label in enumerate(predictions):
            writer.writerow({'ImageId':idx+1, 'Label':label})
            
    if online:
        raise NotImplementedError
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Kaggle auto submit',
        # formatter_class = argparse.RawDescriptionHelpFormatter,
        # epilog = textwrap.dedent('''Example:
        #                         submission.py -m <model_name> -o <boolean> -p <submission_save_path> -v <boolean>
        #                         ''')
        )
    parser.add_argument('--model', type=str, default='model', help='model name')
    parser.add_argument('--online', type=bool, default=False, help='submit to kaggle.com')
    parser.add_argument('--path', type=str, default='datasets/submission/', help='submission file path')
    parser.add_argument('--verbose', type=bool, default=False, help='show status')
    
    args = parser.parse_args()
    submit(args.model, args.online, args.path, args.verbose)