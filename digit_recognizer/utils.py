import pickle

def load_data():
    with open('datasets/data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data