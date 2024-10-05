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


@tf.function
def caller(model, batch):
    outs = model(batch, training=True)
    return outs

@tf.function
def calc_median(x):
    #if x.get_shape()[0] is None:
    #    m = 10
    #else:
    m = x.get_shape()[0] // 2
    med = tf.reduce_min(tf.nn.top_k(x, m, sorted=False).values)
    return med

#@tf.function
def median_loop(model, ds):
    bds = ds.repeat(1)
    numbatches = tf.data.experimental.cardinality(ds).numpy()
    p = Progbar(numbatches)
    r = []
    i = 0
    for x, y in bds:
        outs = caller(model, x)
        #tf.print(outs)
        r.append(outs)
        i += 1
        p.update(i)
    c = tf.concat(r, axis=0)

    #print('median:')
    #c = tf.concat(r2, axis=0)
    vals = tf.transpose(c)
    md = tf.map_fn(calc_median, vals)
    return md

def run():
    #tf.config.run_functions_eagerly(True)

    #a = tf.cond(tf.equal(tf.constant(1), tf.constant(0)), lambda: True, lambda: False)

    trainds, valds = data.load_dataset('images_large', sample_size=128)
    transpath = 'session/trans_feat_full_new2.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    base_model = models.load_model('largeimage16.keras', compile=True)
    base_model.trainable = False

    layermodel = mlutil.make_output_nodes(base_model)

    inputs = keras.Input(base_model.input_shape[1:])
    xb = layermodel(inputs)
    x = [TransactionLayer()(xo) for xo in xb[:-1]]
    x = layers.Concatenate()(x)
    # x = TransformLayer(layer_names=layermodel.output_names, transform='binary_median', name='pat_transform')(x)
    #x = BinarizeLayer(layer_names=layermodel.output_names, name='pat_transform')(x)
    #xd = layers.Identity()(xb[-1])  # will not deserialize without this
    #tmodel = Model(inputs=inputs, outputs=[x, xd])
    tmodel = Model(inputs=inputs, outputs=x)
    o = median_loop(tmodel, trainds)

    x = BinarizeLayer(layer_names=layermodel.output_names, name='pat_transform')(x)
    xd = layers.Identity()(xb[-1])  # will not deserialize without this
    tmodel2 = Model(inputs=inputs, outputs=[x, xd])

    tmodel2.get_layer('pat_transform').set_weights([o])

    tmodel2.save('session/tmodel.keras')
    tmodel3 = models.load_model('session/tmodel.keras', compile=True)

    trans = tmodel3.predict(trainds)
    bdf = transactions_to_dataframe(tmodel2, trans, trainds)




    tmodel = TransactionLayer.make_model(base_model)
    #tmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
    #tmodel.fit(trainds, epochs=1)
    TransactionLayer.train(tmodel, trainds)
    #tmodel.save('session/tmodel.keras')
    #tmodel2 = models.load_model('session/tmodel.keras')


    #preds = base_model.predict(tst)
    trans = tmodel.predict(trainds)
    return
    #trans.to_csv('session/trans.csv')
    #trans2 = tmodel2.predict(trainds)
    #trans2 = None
    #bdf = transactions_to_dataframe(tmodel, trans, trainds)
    #bdf2 = transactions_to_dataframe(tmodel2, trans2, trainds)
    #bdf.to_csv('session/btrans.csv')
    bdf = pd.read_csv('session/btrans.csv', index_col='index')

    minsup = 0.6
    minsupratio = 1.1
    #cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    cpats = pats_by_layer(bdf, 'activation-', 2.0, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    #v = list(cpats[0]['pattern'])
    patternset = list(elems.keys())


    r = layermodel.predict(valds)
    print(1)