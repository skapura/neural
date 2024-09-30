from keras import layers, models
from keras.models import Model
import data
import keras
import mlutil
import tensorflow as tf
import numpy as np
from patterns import feature_activation_max
from trans_model import TransactionLayer

def inner(x):
    print('inner:')
    print(x.shape)
    return x


def test(x):
    x = tf.transpose(x, perm=[2, 0, 1])
    x = tf.map_fn(feature_activation_max, x)
    print(x.shape)
    return x

def run():
    tf.config.run_functions_eagerly(True)

    #a = np.zeros((5, 256, 256, 64), dtype=int)
    #abc = tf.map_fn(test, a)
    my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    max_vals = tf.Variable(my_tensor)


    trainds, valds = data.load_dataset('images_large', sample_size=64)
    transpath = 'session/trans_feat_full_new2.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    base_model = models.load_model('largeimage16.keras', compile=True)

    tmodel = TransactionLayer.make_model(base_model)
    #preds = base_model.predict(tst)
    trans = tmodel.predict(trainds)


    r = layermodel.predict(valds)
    print(1)