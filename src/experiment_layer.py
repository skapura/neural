import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional, Model
import pandas as pd
import os
import numpy as np
import joblib
import shutil
import mlutil
import data
import patterns as pats
import const
from pattern_model import PatternLayer, PatternModel



def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def pats_by_layer(bdf, columns, label, minsup, minsupratio):
    col = [c for c in bdf.columns if columns in c]
    sel = bdf.loc[bdf['label'] == label].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != label].drop(const.META, axis=1)[col]

    #minsup = 0.7
    #minsupratio = 1.1
    imatch, nonmatch = pats.matches(sel, set(['activation_1-8']))
    isup = len(imatch) / len(sel)
    nonsup = len(nonmatch) / len(sel)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    #elems = pats.unique_elements(cpats)
    return cpats


def build_pat_model(trainds, transpath, valds, valpath):
    base_model = models.load_model('largeimage16.keras', compile=True)
    bdf, scaler = pats.preprocess(base_model, trainds, transpath)

    #col = [c for c in bdf.columns if 'activation-' in c]
    #sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    #notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]

    minsup = 0.5
    minsupratio = 1.1
    #cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    cpats = pats_by_layer(bdf, 'activation_1-', 0.0, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    v = list(elems.keys())

    base_model = models.load_model('largeimage16.keras', compile=True)
    pmodel = PatternModel.make(base_model, v, 0)
    pmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    pmodel.fit(trainds, trans_path=transpath, validation_data=valds, val_path=valpath, epochs=1)
    pmodel.save('session/testpatmodel_class.keras')



def run():
    trainds, valds = data.load_dataset('images_large')
    transpath = 'session/trans_feat_full.csv'
    valpath = 'session/vtrans_feat_full.csv'

    build_pat_model(trainds, transpath, valds, valpath)

    pmodel = models.load_model('session/testpatmodel_class.keras', compile=True)

    results = pmodel.evaluate(trainds, transpath)
    print(1)

