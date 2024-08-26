import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
import pandas as pd
from data import load_dataset, scale, get_ranges, image_path, activation_info
import patterns as pats
import mlutil
from plot import output_features, overlay_heatmap
import const
import cv2
import numpy as np


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
    classes = trainds.class_names

    #trans = pats.model_to_transactions(model, outputlayers, trainds)
    #trans.to_csv('session/trans_feat16.csv')
    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    #c, a = activation_info(bdf)
    #a['sortrow0'] = a['support-0'] - (a['support-1'] + a['support-2'])
    #a['sortrow1'] = a['support-1'] - (a['support-0'] + a['support-2'])
    #a['sortrow2'] = a['support-2'] - (a['support-0'] + a['support-1'])
    #b = a[['sortrow0', 'sortrow1', 'sortrow2']].max(axis=1)
    #a['maxsort'] = b
    #sorted = a.sort_values('maxsort')


    #col = [c for c in bdf.columns if 'activation-' in c]
    col = bdf.columns.difference(const.META, sort=False)
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]

    #sel = bdf.loc[bdf['label'] != bdf['predicted']].drop(const.META, axis=1)[col]
    #notsel = bdf.loc[bdf['label'] == bdf['predicted']].drop(const.META, axis=1)[col]

    #sel = bdf.loc[(bdf['label'] != bdf['predicted']) & (bdf['label'] == 0.0)].drop(const.META, axis=1)[col]
    #notsel = bdf.loc[(bdf['label'] == bdf['predicted']) & (bdf['label'] == 0.0)].drop(const.META, axis=1)[col]

    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    targetmatches = set()
    othermatches = set()
    for e in elems.values():
        targetmatches.update(e['targetmatches'])
        othermatches.update(e['othermatches'])
    #cpats = pats.filter_patterns_by_layer(cpats, ['activation_1', 'activation_2', 'activation_3'])
    #mpats2 = pats.filter_patterns_by_layer(mpats, ['activation_1'])
    #path = image_path(trans, 1)
    #imagedata = outputmodel.load_image(path)
    #heat, heatmapimage = mlutil.heatmap(np.asarray([imagedata]), model, 'activation_3')
    #heatover = overlay_heatmap(imagedata, heatmapimage)
    #cv2.imwrite('session_new/heat.png', heatover)

    #filters = outputmodel.filters_in_layer('activation_3')
    #path = image_path(trans, 0)
    #output_features(outputmodel, path, filters, 'session_new/test', franges)

    #churches = trans.loc[trans['label'] == classes.index('church')]
    #for index, _ in churches.iloc[:5].iterrows():
    #    path = image_path(trans, index)
    #    prefix = 'session_new/' + str(index).zfill(4)
    #    filters = outputmodel.filters_in_layer('activation')
    #    output_features(outputmodel, path, filters, prefix, franges)

    #for imageindex in othermatches:
    #    path = image_path(trans, imageindex)
    #    prefix = 'session_new/other/' + str(imageindex).zfill(4)
    #    output_features(outputmodel, path, elems.keys(), prefix, franges)


    for i, p in enumerate(cpats[:1]):
        print(i)
        for j in range(50):
            imageindex = p['targetmatches'][j]
            path = image_path(trans, imageindex)
            prefix = 'session_new/pat_' + str(i) + '-' + str(imageindex).zfill(4)
            output_features(outputmodel, path, p['pattern'], prefix, franges)
    print('test')
