import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from utils import load_data, plot_history

parser = argparse.ArgumentParser()
parser.add_argument('--save_model', type=bool, default=False, help='save model as hdf5 format')
parser.add_argument('--model_name', type=str, default=None, help='name of model')
parser.add_argument('--save_plot', type=bool, default=False, help='save history plot')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch_size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()


X_train, X_val, X_test, y_train, y_val, y_test = load_data(flatten=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (5, 5), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (5, 5), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
], name=args.model_name)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=args.epochs)

if args.save_model:
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model.name}.h5')
    
plot_history(history, fig_name=model.name, save=args.save_plot)