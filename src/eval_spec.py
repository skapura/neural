from keras import models
import pandas as pd
import pickle
import data
import patterns as pats
import const


def load_data():
    return data.load_from_directory('datasets/images_large/train')


if __name__ == '__main__':
    ds = load_data()
    model = models.load_model('session/temp_model.keras', compile=True)
    results = model.evaluate(ds, return_dict=True)
    with open('session/results.pkl', 'wb') as file:
        pickle.dump(results, file)
    print('results: ' + str(results))
