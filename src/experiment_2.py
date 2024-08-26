import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
import pandas as pd
from data import load_dataset, load_dataset_selection, get_ranges, scale
import patterns as pats
import mlutil
import const

from keras.src.utils import dataset_utils
from keras.src.backend.config import standardize_data_format
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
import numpy as np


class PatternBranch(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('init pattern')

    def call(self, inputs):
        return inputs


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = PatternBranch()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def build_model_sel():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    #x = PatternBranch()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def run():

    trainds, valds = load_dataset('images_large')

    #model = build_model()
    model = models.load_model('largeimage16.keras', compile=True)
    #w = oldmodel.get_weights()
    #model.set_weights(w)
    #test_loss, test_acc = model.evaluate(trainds)
    #test_loss_old, test_acc_old = oldmodel.evaluate(trainds)

    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    col = bdf.columns.difference(const.META, sort=False)
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]
    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    p = cpats[0]

    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    #vtrans = pats.model_to_transactions(model, outputlayers, valds)
    #vtrans.to_csv('session/vtrans_feat16.csv')
    vtrans = pd.read_csv('session/vtrans_feat16.csv', index_col='index')
    scaled = scale(vtrans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    imatch, _ = pats.matches(bdf.drop(const.META, axis=1), p['pattern'])
    labels = vtrans.iloc[imatch]['label']
    midx = list(labels[labels == 0].index.values)
    nidx = list(labels[labels != 0].index.values)
    t = vtrans.iloc[midx]['path'].to_list()
    o = vtrans.iloc[nidx]['path'].to_list()
    valds_sel = load_dataset_selection('images_large/val', selection=(t, o), label_mode='categorical')
    valds_sel2 = load_dataset_selection('images_large/val', selection=t + o, label_mode='categorical')
    selmodel = models.load_model('submodel_cat.keras')
    model = models.load_model('largeimage16.keras')
    loss_sel, acc_sel = selmodel.evaluate(valds_sel)
    ls, acc = model.evaluate(valds_sel2)

    t = trans.iloc[p['targetmatches']]['path'].to_list()
    o = trans.iloc[p['othermatches']]['path'].to_list()
    trainds_sel = load_dataset_selection('images_large/train', selection=(t, o), label_mode='categorical')
    #acc_sel, loss_sel = selmodel.evaluate(trainds_sel)
    #acc, ls = model.evaluate(trainds2)

    submodel = build_model()
    submodel.fit(trainds_sel, validation_data=valds_sel, epochs=10)
    submodel.save('submodel_cat.keras')

    print(1)
