import tensorflow as tf
from keras import layers
from keras.src.models import Model
from keras import models
from keras.src.utils import Progbar
from zipfile import ZipFile
import os
import mlutil
import models.trans_model as tm
import data as data


# Expects binary input
class MatchLayer(layers.Layer):
    def __init__(self, pattern_index_set, **kwargs):
        super().__init__(**kwargs)
        self.pattern_set_index = pattern_index_set

    def compute_output_shape(self, input_shape):
        return [None, 1]

    @tf.function
    def call(self, inputs):
        matcharray = tf.TensorArray(tf.bool, size=len(self.pattern_set_index), dynamic_size=False)
        for i, pi in enumerate(self.pattern_set_index):
            selected = tf.gather(inputs, indices=pi, axis=1)
            matches = tf.map_fn(tf.reduce_all, selected)
            matcharray = matcharray.write(i, matches)
        m = matcharray.stack()
        matchany = tf.map_fn(tf.reduce_any, tf.transpose(m, perm=[1, 0]))
        return matchany


class SelectionLayer(layers.Layer):
    def __init__(self, pat_index, **kwargs):
        super().__init__(**kwargs)
        self.pat_index = pat_index

    def call(self, inputs):
        selected = tf.gather(inputs, indices=self.pat_index, axis=3)
        return selected


class BranchLayer(layers.Layer):
    def __init__(self, pat_pred, base_pred, **kwargs):
        super().__init__(**kwargs)
        self.pat_pred = pat_pred
        self.base_pred = base_pred

    def compute_output_shape(self, input_shape):
        return [(None, 1), self.pat_pred.output_shape, (None, 1), self.base_pred.output_shape]

    @tf.function
    def call(self, inputs):
        feats, selectedfeats, matches = inputs
        # mergedpreds = tf.TensorArray(self.base_pred.output[0].dtype, dynamic_size=False, size=tf.size(matches))
        matchidx = tf.reshape(tf.where(matches), shape=[-1])
        nonmatchidx = tf.reshape(tf.where(tf.math.logical_not(matches)), shape=[-1])

        # Predict pattern matches
        patpreds = tf.reshape((), (0, self.pat_pred.output_shape[-1]))
        if not tf.equal(tf.size(matchidx), 0):
            fmatches = tf.gather(selectedfeats, indices=matchidx, axis=0)
            patpreds = self.pat_pred(fmatches)
            # mergedpreds = mergedpreds.scatter(matchidx, patpreds)

        # Predict non-matches
        basepreds = tf.reshape((), (0, self.base_pred.output_shape[-1]))
        if not tf.equal(tf.size(nonmatchidx), 0):
            fnonmatches = tf.gather(feats, indices=nonmatchidx, axis=0)
            basepreds = self.base_pred(fnonmatches)
            # mergedpreds = mergedpreds.scatter(nonmatchidx, basepreds)

        # preds = mergedpreds.stack()
        return matchidx, patpreds, nonmatchidx, basepreds


def patterns_to_index(patterns, features):
    pat_indexes = []
    for pat in patterns:
        pat_index = [features.index(p) for p in pat]
        pat_indexes.append(pat_index)
    return pat_indexes


def match_dataset(pmodel, ds, binary_target=None):
    matchmodel = Model(inputs=pmodel.inputs, outputs=pmodel.get_layer('pat_matches').output)
    offset = 0
    allmatches = []
    p = Progbar(tf.data.experimental.cardinality(ds).numpy())
    for i, (x, y) in enumerate(ds):
        matches = matchmodel(x)
        matches = tf.reshape(tf.where(matches), shape=[-1])
        matches = tf.add(matches, offset)
        allmatches += matches.numpy().tolist()
        offset += len(x)
        p.update(i + 1)
    sfiles = [ds.file_paths[i] for i in allmatches]
    slabels = [ds.labels[i] for i in allmatches]
    sds = data.image_subset(sfiles, slabels, ds.class_names, binary_target=binary_target)
    return sds


def build_pattern_predictor(input_shape):
    inputs = layers.Input(input_shape)
    x = layers.MaxPooling2D((2, 2), name='here')(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    dense = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    # dense = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Model(inputs=inputs, outputs=dense, name='pat_pred')
    # patpred = Model(inputs=inputs, outputs=dense3, name='pat_pred')
    return model


def save_pattern_model(pmodel, path):
    d = {'patterns': [list(l) for l in pmodel.patterns]}
    data.save(d, 'temp/p_data.pickle')
    pmodel.base_model.save('temp/p_base.keras')
    pmodel.trans_model.save('temp/p_trans.keras')
    pmodel.save_weights('temp/p_model.weights.h5')
    with ZipFile(path, 'w') as z:
        z.write('temp/p_data.pickle', arcname='p_data.pickle')
        z.write('temp/p_base.keras', arcname='p_base.keras')
        z.write('temp/p_trans.keras', arcname='p_trans.keras')
        z.write('temp/p_model.weights.h5', arcname='p_model.weights.h5')
    os.remove('temp/p_data.pickle')
    os.remove('temp/p_base.keras')
    os.remove('temp/p_trans.keras')
    os.remove('temp/p_model.weights.h5')


def load_pattern_model(path=''):
    with ZipFile(path, 'r') as z:
        z.extractall(path='temp')
    d = data.load('temp/p_data.pickle')
    base_model = models.load_model('temp/p_base.keras', compile=False)
    trans_model = models.load_model('temp/p_trans.keras')
    pmodel = build_pattern_model(d['patterns'], base_model, trans_model)
    pmodel.load_weights('temp/p_model.weights.h5')
    os.remove('temp/p_data.pickle')
    os.remove('temp/p_base.keras')
    os.remove('temp/p_trans.keras')
    os.remove('temp/p_model.weights.h5')
    return pmodel


# Assumes all pattern elements are from same layer
def build_pattern_model(patterns, base_model, trans_model, skiptrain=False):
    #trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)
    base_model.Trainable = False
    baseclone = models.clone_model(base_model)

    # Feature extraction
    patlayername = mlutil.parse_feature_ref(patterns[0][0])[0]
    featextract = tm.build_feature_extraction(base_model, [patlayername], False)
    #featextract2, _ = tm.build_feature_extraction(base_model, [patlayername], True)

    # Base prediction
    patlayer = base_model.get_layer(patlayername)
    pi = base_model.layers.index(patlayer) + 1
    basepred = Model(inputs=base_model.layers[pi].input, outputs=base_model.output, name='base_pred')

    # Pattern matcher
    blayer = trans_model.get_layer('pat_binarize')
    pidx = patterns_to_index(patterns, blayer.feature_names)
    pfeats = list(set(sum(pidx, [])))
    pfeats.sort()

    # Pattern model
    inputs = layers.Input(featextract.input_shape[1:])
    xf = featextract(inputs)
    xb = blayer(xf[-1])
    xm = MatchLayer(pidx, name='pat_matches')(xb)
    xp = SelectionLayer(pfeats)(xf[0])
    pat_pred = build_pattern_predictor(xp.shape[1:])
    x = BranchLayer(pat_pred, basepred)([xf[0], xp, xm])
    pmodel = Model(inputs=inputs, outputs=x)

    # Pattern trainer
    inputs = layers.Input(featextract.input_shape[1:])
    xf = featextract(inputs)
    featextract.Trainable = False
    xp = SelectionLayer(pfeats)(xf[0])
    x = pat_pred(xp)
    ptmodel = Model(inputs=inputs, outputs=x)
    ptmodel.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=['accuracy'])

    pmodel.patterns = patterns
    pmodel.base_model = baseclone
    pmodel.trans_model = trans_model
    pmodel.train_model = ptmodel
    #pmodel.pat_pred = pat_pred

    return pmodel


def train_pattern_model(pmodel, trainds):

    pds = match_dataset(pmodel, trainds, binary_target='church')
    #data.save_subset(pds, 'session/testds.pickle')
    #pds2 = data.load_subset('session/testds.pickle')

    pmodel.train_model.fit(pds, epochs=2)

    # inputs = layers.Input(trans_model.input_shape[1:])
    # x = trans_model(inputs)
    # x = MatchLayer(pidx)(x[0])
    # model = Model(inputs=inputs, outputs=x)

    print(1)
