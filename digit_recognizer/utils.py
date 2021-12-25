import os
import pickle

import matplotlib.pyplot as plt

def load_data(flatten=True):
    with open('datasets/data.pkl', 'rb') as f:
        data = pickle.load(f)
    if flatten:
        return data
    else:
        for i in range(3):
            data[i] = (data[i]/255.0).reshape(-1, 28, 28, 1)
        return data

def plot_history(history,  fig_name, save=False):
    train_history = history.history['loss']
    validation_history = history.history['val_loss']
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.title('Loss History')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS Function')
    plt.plot(train_history, 'red', label='train')
    plt.plot(validation_history, 'blue', label='validation')
    plt.grid(True)
    plt.legend()

    train_history = history.history['accuracy']
    validation_history = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.title('Accuracy History')
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(train_history, 'red', label='train')
    plt.plot(validation_history, 'blue', label='validation')
    plt.grid(True)
    plt.legend()
    
    if save:
        os.makedirs('train_plots', exist_ok=True)
        fig.savefig(f"train_plots/{fig_name}_history.png")

    plt.show()