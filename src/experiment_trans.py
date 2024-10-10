from keras import layers, models
from keras.models import Model
from keras.src.utils import Progbar
import data
import keras
import mlutil
import tensorflow as tf
import numpy as np
import pandas as pd
from patterns import feature_activation_max
import patterns as pats
import const
from trans_model import TransactionLayer, transactions_to_dataframe, BinarizeLayer
from pattern_model import PatternLayer, PatternSelect, PatternMatch #, PatternBranch


def pats_by_layer(bdf, columns, label, minsup, minsupratio):
    col = [c for c in bdf.columns if columns in c]
    sel = bdf.loc[bdf['label'] == label].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != label].drop(const.META, axis=1)[col]
    print('target # instances: ' + str(len(sel)))
    print('other # instances: ' + str(len(notsel)))
    #minsup = 0.7
    #minsupratio = 1.1
    #imatch, nonmatch = pats.matches(sel, set(['activation_1-8']))
    #isup = len(imatch) / len(sel)
    #nonsup = len(nonmatch) / len(sel)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    #elems = pats.unique_elements(cpats)
    return cpats

def build_pre_model(base_model, trainds):
    tmodel = BinarizeLayer.train(base_model, trainds)
    tmodel.save('session/tmodel.keras')
    #tmodel2 = models.load_model('session/tmodel.keras', compile=True)
    trans = tmodel.predict(trainds)
    bdf = transactions_to_dataframe(tmodel, trans, trainds)
    bdf.to_csv('session/btrans.csv')
    return tmodel, bdf


def run():
    tf.config.run_functions_eagerly(True)

    #a = tf.TensorArray(tf.float32, size=32, dynamic_size=True)
    #a.write(0, tf.constant([1.0, 2.0, 3.0]))
    #a.write(1, tf.constant([4.0, 5.0, 6.0]))
    #b = a.stack()

    base_model = models.load_model('largeimage16.keras', compile=True)
    trainds, valds = data.load_dataset('images_large')#, sample_size=128)
    transpath = 'session/trans_feat_full_new2.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    #build_pre_model(trainds)
    tmodel = models.load_model('session/tmodel.keras', compile=True)
    bdf = pd.read_csv('session/btrans.csv', index_col='index')
    #print(bdf)
    #return

    minsup = 0.7
    minsupratio = 1.1
    #cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    cpats = pats_by_layer(bdf, 'activation-', 0.0, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    #v = list(cpats[0]['pattern'])
    patternset = list(elems.keys())

    pattern = cpats[1]['pattern']
    pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in pattern]

    inputs = keras.Input(shape=base_model.input_shape[1:])
    medians = tmodel.get_layer('pat_transform').medians
    x = PatternLayer(pattern, 0.0, medians, base_model)(inputs)
    pmodel = Model(inputs=inputs, outputs=x)
    for step, (x_batch, y_batch) in enumerate(trainds):
        a = pmodel(x_batch)
        print(1)

    player = base_model.get_layer('activation')
    feat_extract = Model(inputs=base_model.inputs, outputs=player.output, name='feat_extract')
    inputs = keras.Input(shape=feat_extract.output_shape[1:])
    x = PatternSelect(pattern_feats)(inputs)
    xt = TransactionLayer()(x)
    medians = tmodel.get_layer('pat_transform').medians
    selmedians = tf.gather(medians, indices=pattern_feats, axis=0)
    x = PatternMatch(medians=selmedians)(xt)
    x = PatternBranch()([xt, x])
    patselect = Model(inputs=inputs, outputs=x)

    for step, (x_batch, y_batch) in enumerate(trainds):
        outs = feat_extract(x_batch)
        selouts = patselect(outs)
        #a = tf.reduce_any(selouts)
        print(step)
        print(selouts)

    x = PatternLayer(pattern, 0.0, base_model)(inputs)
    pmodel = Model(inputs=inputs, outputs=x)
    pmodel.save('session/patmodel.keras')
    pmodel2 = models.load_model('session/patmodel.keras', compile=True)


    print(1)