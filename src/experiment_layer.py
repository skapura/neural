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
from pattern_model import PatternSelect, PatternMatch, PatternLayer, PatternModel



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

    minsup = 0.7
    minsupratio = 1.1
    #cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    cpats = pats_by_layer(bdf, 'activation-', 0.0, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    v = list(cpats[0]['pattern'])
    #v = list(elems.keys())

    base_model = models.load_model('largeimage16.keras', compile=True)
    pmodel = PatternModel.make(base_model, v, 0)
    pmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'], run_eagerly=True)
    pmodel.fit(trainds, trans_path=transpath, validation_data=valds, val_path=valpath, epochs=1)
    pmodel.save('session/testpatmodel_class.keras')
    return pmodel


def run():
    #tf.config.run_functions_eagerly(True)
    trainds, valds = data.load_dataset('images_large')
    transpath = 'session/trans_feat_full_new.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    #model = models.load_model('session/test.keras', compile=True)
    #model.evaluate(valds)

    # Feature extract
    base_model = models.load_model('largeimage16.keras', compile=True)

    bdf, scaler = pats.preprocess(base_model, trainds, transpath)

    inputs = keras.Input(base_model.input_shape[1:])
    x = PatternLayer(['activation_1-4', 'activation_1-5', 'activation_1-6'], 0, base_model, scaler)(inputs)
    p = PatternModel(inputs=inputs, outputs=x)
    p.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'], run_eagerly=True)
    p.fit(trainds, trans_path=transpath, epochs=1)
    p.trainable = True
    #p.save('session/test.keras')
    #p2 = models.load_model('session/test.keras')
    r = p.evaluate(trainds)


    player = base_model.get_layer('activation_1')
    fextract = Model(inputs=base_model.inputs, outputs=player.output)
    #fextract.save('session/fextract.keras')
    #fextract2 = models.load_model('session/fextract.keras')

    # Base Predict
    li = base_model.layers.index(player) + 1
    binputs = keras.Input(shape=base_model.layers[li].input.shape[1:])
    x = base_model.layers[li](binputs)
    for i in range(li + 1, len(base_model.layers)):
        x = base_model.layers[i](x)
    base_pred = Model(inputs=binputs, outputs=x, name='base_pred')
    #base_pred = Model(inputs=base_model.layers[li].input, outputs=base_model.output)

    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    first = Model(inputs, x)

    inputs = keras.Input(shape=first.output_shape[1:])
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    second = Model(inputs, x)

    inputs = keras.Input(shape=(256, 256, 3))
    x = first(inputs)
    x = second(x)
    combined = Model(inputs, x)
    #combined.save('session/test.keras')
    #combined2 = models.load_model('session/test.keras')

    first.save('session/first.keras')
    second.save('session/second.keras')
    first2 = models.load_model('session/first.keras')
    second2 = models.load_model('session/second.keras')
    inputs = keras.Input(shape=first2.input_shape[1:])
    x = first2(inputs)
    x = second2(x)
    combined2 = Model(inputs, x)




    # Pat Predict
    pat = [0, 1, 5]
    inputs = keras.Input(shape=fextract.output_shape[1:])
    #x = PatternSelect(pat)(pinputs)
    x = layers.MaxPooling2D((2, 2))(inputs)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    pat_pred = Model(inputs=inputs, outputs=x)

    # Pat Train
    inputs = keras.Input(shape=(256, 256, 3))
    x = fextract(inputs, training=False)
    #x = pat_pred.layers[1](x)
    x = pat_pred(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(3, name='prediction', activation='softmax')(x)
    pat_train = Model(inputs=inputs, outputs=x)
    pat_train.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    pat_train.save('session/test.keras')
    model2 = models.load_model('session/test.keras')

    # Main branch model
    inputs = keras.Input(shape=(256, 256, 3))
    #x = fextract(inputs, training=False)
    xp = PatternMatch(pat)(fextract.output)
    x = Branch(pat, base_pred, pat_pred)([x, xp])
    model = Model(inputs=fextract.input, outputs=x)
    #model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #              metrics=['accuracy'])
    model.save('session/test.keras')
    model2 = models.load_model('session/test.keras')
    print(1)

    #pat_train.fit(trainds, epochs=1)
    #r = pat_train.evaluate(valds)

    #fextract.save('session/fextract.keras')
    #base_pred.save('session/basepred.keras')
    #pat_pred.save('session/patpred.keras')
    #pat_train.save('session/pattrain.keras')
    #model.save('session/test.keras')

    #fextract2 = models.load_model('session/fextract.keras')
    #base_pred2 = models.load_model('session/basepred.keras')
    #pat_pred2 = models.load_model('session/patpred.keras')
    #pat_train2 = models.load_model('session/pattrain.keras')
    model2 = models.load_model('session/test.keras')

    print(1)


    #x = layers.Rescaling(1.0 / 255)(inputs)
    #x = layers.Conv2D(128, (3, 3))(x)
    #x = layers.Activation("relu")(x)
    #x = PatternSel([0, 1, 5])(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(64, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Conv2D(128, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(3, name='prediction', activation='softmax')(x)
    #model = Functional(inputs, x)
    #model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #              metrics=['accuracy'])


    #tf.config.run_functions_eagerly(True)


    model.fit(trainds, epochs=1)
    model.save('session/test.keras')

    #pmodel = build_pat_model(trainds, transpath, valds, valpath)

    pmodel = models.load_model('session/testpatmodel_class.keras', compile=True)

    #bresults = pmodel.pat_layer.base_model.evaluate(trainds, return_dict=True)
    #pmodel.evaluate2(trainds, transpath)
    y2 = np.concatenate([y for _, y in valds], axis=0)
    presults = pmodel.evaluate(valds)

    print(1)

