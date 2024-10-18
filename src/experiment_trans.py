from keras import layers, models
from keras.models import Model
from keras.src.utils import Progbar
from keras.src.models import Functional
import data
import keras
import mlutil
import tensorflow as tf
import numpy as np
import pandas as pd
from patterns import feature_activation_max
import patterns as pats
import const
from azure import AzureSession
from trans_model import transactions_to_dataframe, build_transaction_model, fit_medians #, TransactionLayer, transactions_to_dataframe, BinarizeLayer
from pattern_model import build_pattern_model, pat_evaluate#PatternLayer, PatternSelect, PatternMatch, PatModel
#from pattern_model import PatternMatch


def pats_by_layer(bdf, columns, label, minsup, minsupratio):
    col = [c for c in bdf.columns if columns in c]
    sel = bdf.loc[bdf['label'] == label].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != label].drop(const.META, axis=1)[col]
    print('target # instances: ' + str(len(sel)))
    print('other # instances: ' + str(len(notsel)))
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    return cpats

def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3))(x)
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


def train_base_model(trainds, model_path, epochs):
    base_model = build_model()
    base_model.fit(trainds, epochs=epochs)
    base_model.save(model_path)
    return base_model




def build_pre_model(base_model, trainds):
    tmodel = BinarizeLayer.train(base_model, trainds)
    tmodel.save('session/tmodel.keras')
    #tmodel2 = models.load_model('session/tmodel.keras', compile=True)
    trans = tmodel.predict(trainds)
    bdf = transactions_to_dataframe(tmodel, trans, trainds)
    bdf.to_csv('session/btrans.csv')
    return tmodel, bdf


def matching_dataset(pt, ds):
    featextract = pt.layers[-1].feat_extract
    matcher = pt.layers[-1].pat_matcher
    allmatches = list()
    for step, (x, y) in enumerate(ds):
        print(step)
        feats = featextract(x)
        matches = matcher(feats[1])
        midx = tf.squeeze(tf.where(matches))
        midx = tf.cast(midx, dtype=tf.int32)
        midx += step * len(x)
        allmatches += midx.numpy().tolist()
        #allmatches.append(midx)
    print(1)

def test():
    trainds = data.load_from_directory('datasets/' + 'images_large' + '/train')
    valds = data.load_from_directory('datasets/' + 'images_large' + '/val')

    # Base model
    basepath = 'session/base.keras'
    base_model = train_base_model(trainds, basepath, 30)
    base_model = models.load_model(basepath, compile=True)


    patlayers = ['activation']
    #base_model = build_model()
    #base_model.fit(trainds, epochs=30)
    #evalb = base_model.evaluate(trainds, return_dict=True)
    #evalv = base_model.evaluate(valds, return_dict=True)
    #base_model.save('session/base.keras')
    #base_model = models.load_model('largeimage_small.keras', compile=True)
    base_model = models.load_model('session/base.keras', compile=True)
    tmodel = build_transaction_model(base_model, trainds, patlayers)
    tmodel.save('session/tmodel.keras')
    #tmodel = models.load_model('session/test.keras', compile=True)

    trans = tmodel.predict(trainds)
    bdf = transactions_to_dataframe(tmodel, trans, trainds)
    bdf.to_csv('session/bdf.csv')
    bdf = pd.read_csv('session/bdf.csv', index_col='index')

    minsup = 0.5
    minsupratio = 1.1
    cpats = pats_by_layer(bdf, 'activation-', 0.0, minsup, minsupratio)
    pattern = cpats[1]

    matches = pattern['targetmatches'] + pattern['othermatches']
    images = [trainds.file_paths[p] for p in matches]
    labels = [trainds.labels[p] for p in matches]
    subds = data.image_subset(images, labels, trainds.class_names, binary_target=0)

    pmodel, ptrain = build_pattern_model(pattern['pattern'], base_model, tmodel)
    #sres = ptrain.evaluate(subds, return_dict=True)
    ptrain.fit(subds, epochs=30)
    #ptrain.layers[-1].pat_pred.save_weights('session/ptrain.weights.h5')
    #ptrain.layers[-1].pat_pred.load_weights('session/ptrain.weights.h5')
    #ptrain.save('session/ptrain.keras')
    #sresa = ptrain.evaluate(subds, return_dict=True)

    #ptpreds = pmodel.evaluate(trainds, return_dict=True)
    eval_pat_trn = ptrain.evaluate(subds)
    eval_pat = pat_evaluate(pmodel, trainds)
    eval_base = base_model.evaluate(trainds, return_dict=True)

    print(ptpreds)
    print(basepreds)


@tf.function
def scattertest():
    a = tf.constant([[1, 2], [3, 4]])
    idx = tf.constant([1, 3])
    ta = tf.TensorArray(tf.int32, size=5)
    ta = ta.scatter(idx, a)
    c = ta.stack()
    #tf.map_fn(lambda x: x + 1, a)
    return c


def remote_train(model, session):
    model.save('session/temp_model.keras')
    session.put('session/temp_model.keras', dest='neural/session/temp_model.keras')
    session.execute('cd neural')
    session.execute('python src/train_spec.py 3')
    session.get('neural/session/temp_model.keras', dest='session/temp_model.keras')
    trained = models.load_model('session/temp_model.keras', compile=True)
    return trained


def run():
    #tf.config.run_functions_eagerly(True)


    base_model = build_model()
    sess = AzureSession()
    sess.open()
    remote_train(base_model, sess)
    #sess.execute('ls -l')
    #sess.execute('cat gpu_test.py')
    sess.close()


    a = tf.constant([[1], [2], [3]])
    o = tf.constant([[5]])
    b = tf.squeeze(a)
    bo = tf.squeeze(o)
    b2 = tf.reshape(a, shape=[-1])
    bo2 = tf.reshape(o, shape=[-1])

    test()
    return



    #trainds = data.load_from_directory('datasets/' + 'images_large' + '/train')
    trainds = data.load_from_directory('datasets/' + 'images_large' + '/train')
    #img = trainds.file_paths[:32]
    #lbl = trainds.labels[:32]
    #trainds2 = data.image_subset(img, lbl, trainds.class_names, binary_target=None)

    #base_model = models.load_model('largeimage16.keras', compile=True)
    base_model = models.load_model('largeimage_small.keras', compile=True)
    #base_model = build_model()
    #base_model.fit(trainds, epochs=2)
    #base_model.save('largeimage_small.keras')
    #trainds, valds = data.load_dataset('images_large')#, sample_size=128)
    transpath = 'session/trans_feat_full_new2.csv'
    valpath = 'session/vtrans_feat_full_new.csv'

    #s = [trainds.file_paths[0], trainds.file_paths[2]]
    #data.load_dataset_selection(trainds, selection=s, label_mode='binary')


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
    medians = tmodel.get_layer('pat_transform').medians
    pattern_medians = tf.gather(medians, indices=pattern_feats, axis=0)

    inputs = keras.Input(shape=base_model.input_shape[1:])
    matcher = PatternMatch(tmodel)(inputs)
    pmodel = Model(inputs=inputs, outputs=matcher)

    pmodel.predict(trainds)



    inputs = keras.Input(shape=base_model.input_shape[1:])
    medians = tmodel.get_layer('pat_transform').medians
    x = PatternLayer(pattern, 0.0, medians, base_model)(inputs)
    pmodel = PatModel(inputs=inputs, outputs=x)
    pmodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])
    r = pmodel.fit(trainds)
    #pmodel.save('session/pmodel.keras')
    #pmodel = models.load_model('session/pmodel.keras', compile=True)
    #pmodel.compile()
    #print(pmodel.metrics)
    #return
    pres = pmodel.evaluate(trainds, return_dict=True)
    bres = base_model.evaluate(trainds, return_dict=True)
    print(pres)
    print(bres)
    return

    numbatches = tf.data.experimental.cardinality(trainds).numpy()
    p = Progbar(numbatches)
    m = keras.metrics.CategoricalAccuracy()
    l = keras.losses.CategoricalCrossentropy(from_logits=False)
    lv = 0.0
    for step, (x_batch, y_batch) in enumerate(trainds):
        a = pmodel(x_batch)
        m.update_state(y_batch, a)
        lv += l(y_batch, a)
        #e = pmodel.evaluate(trainds)
        #print(step)
        p.update(step + 1)
    print(m.result())
    print(lv/ numbatches)
    return
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