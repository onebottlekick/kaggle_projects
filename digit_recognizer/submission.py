import csv
import os
import pickle

import tensorflow as tf

def submit(model_name):
    with open('datasets/submission_data', 'rb') as f:
        data = pickle.load(f)
    data = data.reshape(-1, 28, 28, 1)
    data = data/255.0
    model_path = os.path.join('models', model_name)
    model = tf.keras.models.load_model(model_path + '.h5')
    
    predictions = model.predict(data).argmax(axis=-1)
    
    with open(f'./datasets/submission/{model_name}.csv', 'w', newline='') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, label in enumerate(predictions):
            writer.writerow({'ImageId':idx+1, 'Label':label})
            
submit('hi')