import azure
import tensorflow as tf
from keras.src.utils import dataset_utils
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
from keras.src.backend.config import standardize_data_format
import keras
from keras import layers
from keras.src.models import Functional
import numpy as np
import azure



def run():
    model = azure.train()
    model.save('models/test.keras')
    model = build_model()
    model.save('.untrained.keras')
    #azure.put('activation_binned.csv', dest='Desktop/test.csv')
    #azure.get('abc.txt')
    azure.execute('python3 gpu.py')
    print(1)