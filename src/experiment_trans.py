from keras import layers, models
from keras.models import Model
import data
import keras
import mlutil
import tensorflow as tf
import numpy as np
import pandas as pd
from patterns import feature_activation_max
import patterns as pats
import const
from trans_model import TransactionLayer, transactions_to_dataframe


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


def run():
    tf.config.run_functions_eagerly(True)

    trainds, valds = data.load_dataset('images_large', sample_size=64)
    transpath = 'session/trans_feat_full_new2.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    base_model = models.load_model('largeimage16.keras', compile=True)
    #layermodel = mlutil.make_output_model2(base_model)
    #layermodel.save('session/layer.keras')
    #layermodel2 = models.load_model('session/layer.keras')


    tmodel = TransactionLayer.make_model(base_model)
    labels = TransactionLayer.train(tmodel, trainds)
    #tmodel.save('session/tmodel.keras')
    #tmodel2 = models.load_model('session/tmodel.keras')


    #preds = base_model.predict(tst)
    #trans = tmodel.predict(trainds)
    trans2 = tmodel.predict(trainds)
    #trans2 = None
    bdf = transactions_to_dataframe(tmodel, trans2, trainds)
    #bdf.to_csv('session/btrans.csv')
    bdf = pd.read_csv('session/btrans.csv', index_col='index')

    minsup = 0.7
    minsupratio = 1.1
    #cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    cpats = pats_by_layer(bdf, 'activation_2-', 0.0, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    #v = list(cpats[0]['pattern'])
    patternset = list(elems.keys())


    r = layermodel.predict(valds)
    print(1)