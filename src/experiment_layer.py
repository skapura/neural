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
from pattern_model import PatternLayer, evaluate, PatternModel



def build_model():
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
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def build_pat_model(trainds, valds):
    base_model = models.load_model('largeimage16.keras', compile=True)
    player = PatternLayer(['activation-1', 'activation-2'], 1)
    player.build_branch(base_model)
    x = player(base_model.input)
    pmodel = Model(inputs=base_model.input, outputs=x)
    pmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    player.fit(trainds, validation_data=valds, epochs=1)
    #pmodel.save('session/testpatmodel.keras')
    return pmodel

def run():
    trainds, valds = data.load_dataset('images_large')
    base_model = models.load_model('largeimage16.keras', compile=True)

    pmodel = PatternModel.make(base_model, ['activation-1', 'activation-2'], 1)

    pmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    #pmodel.fit(trainds, validation_data=valds, epochs=1)
    #pmodel.save('session/testpatmodel_class.keras')

    pmodel2 = models.load_model('session/testpatmodel_class.keras', compile=True)

    pmodel2.evaluate(trainds, 'session/trans_feat_full.csv')
    print(1)


def run2():
    trainds, valds = data.load_dataset('images_large')


    #podel = build_pat_model(trainds, valds)
    pmodel = models.load_model('session/testpatmodel.keras', compile=True)
    #tf.config.run_functions_eagerly(True)

    evaluate(pmodel, trainds)
    pmodel.save('session/testpatmodel.keras')
    pmodel2 = models.load_model('session/testpatmodel.keras')

    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    # franges = get_ranges(trans, zeromin=True)
    scaled = data.scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    # col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation-' in c]
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]

    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems = pats.unique_elements(cpats)

    v = list(elems.keys())
    build_branch_model(model, 'activation', [{'filters': v, 'class': 0}])
    #pmodel = models.load_model('session/patmodel.keras')
    #player = pmodel.get_layer('pattern_branch')
    #player.build_branch([{'filters': v, 'class': 0}])
    #pmodel.save('session/patmodel.keras')
    # loaded = models.load_model('session/patmodel.keras')
    print(1)
