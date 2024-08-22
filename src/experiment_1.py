import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
import pandas as pd
from data import load_dataset, scale, get_ranges, image_path
import patterns as pats
import mlutil
from plot import plot_features
import const


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


def run():
    model = models.load_model('largeimage16.keras', compile=True)
    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    last = 'activation_3'
    info = mlutil.conv_layer_info(model, last)
    selectedlayers = mlutil.layer_subset(outputlayers, last)
    outputmodel = mlutil.make_output_model(model, selectedlayers)

    trainds, valds = load_dataset('images_large')
    #trans = pats.model_to_transactions(model, outputlayers, trainds)
    #trans.to_csv('session/trans_feat16.csv')
    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    col = [c for c in bdf.columns if '_1' in c]
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]
    minsup = 0.2
    minsupratio = 2.0
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)

    pattern = cpats[0]
    path = image_path(trans, pattern['targetmatches'][0])
    plot_features(outputmodel, path, pattern['pattern'], franges)
    print('test')
