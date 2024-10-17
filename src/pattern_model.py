import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Model
from keras.src.utils import Progbar
from trans_model import build_feature_extraction
import mlutil


class MatchLayer(layers.Layer):
    def __init__(self, binarizer, pattern, **kwargs):
        super().__init__(**kwargs)
        self.binarizer = binarizer
        f = self.binarizer.feature_names
        self.pat_index = [f.index(p) for p in pattern]

    def compute_output_shape(self, input_shape):
        return [None, 1]

    def call(self, inputs):
        bouts = self.binarizer(inputs)
        selected = tf.gather(bouts, indices=self.pat_index, axis=1)
        matches = tf.map_fn(tf.reduce_all, selected)
        return matches


class BinaryToCategorical(layers.Layer):
    def __init__(self, pattern_class, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.pattern_class = pattern_class
        self.num_classes = num_classes

    def compute_output_shape(self, input_shape):
        return [None, self.num_classes]

    def convert(self, x):
        o = (1.0 - x) / (self.num_classes - 1.0)
        r = tf.stack([x if i == self.pattern_class else o for i in range(self.num_classes)])
        return tf.squeeze(r)

    def call(self, inputs):
        cats = tf.map_fn(self.convert, inputs)
        return cats


class PatternBranch(layers.Layer):
    def __init__(self, feat_extract, pat_matcher, base_pred, pat_pred, **kwargs):
        super().__init__(**kwargs)
        self.feat_extract = feat_extract
        self.pat_matcher = pat_matcher
        self.base_pred = base_pred
        self.pat_pred = pat_pred
        self.categorizer = BinaryToCategorical(0, 3)
        self.matchcount = tf.Variable(0)
        self.matchpatcount = tf.Variable(0)
        self.matchbasecount = tf.Variable(0)
        self.nonmatchcount = tf.Variable(0)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return [None, 3]

    @tf.function
    def call(self, inputs):
        self.matchcount.assign(0)
        self.matchpatcount.assign(0)
        self.matchbasecount.assign(0)
        self.nonmatchcount.assign(0)
        feats = self.feat_extract(inputs)

        # Separate inputs matching pattern
        mergedpreds = tf.TensorArray(inputs.dtype, dynamic_size=True, size=0)
        matches = self.pat_matcher(feats[1])
        midx = tf.cast(tf.reshape(tf.where(matches), shape=[-1]), dtype=tf.int32)
        if not tf.equal(tf.size(midx), 0):
            self.matchcount.assign(tf.size(midx))
            patpredsarray = tf.TensorArray(inputs.dtype, dynamic_size=True, size=0)
            fmatches = tf.gather(feats[0], indices=midx, axis=0)
            fmatches.set_shape([None] + feats[0].shape[1:])
            selectedfeats = tf.gather(fmatches, indices=self.pat_matcher.pat_index, axis=3)
            patbinpreds = self.pat_pred(selectedfeats)

            # Collect matching predictions classified as target class
            pidx = tf.cast(tf.reshape(tf.where(tf.squeeze(patbinpreds) >= 0.5), shape=[-1]), dtype=tf.int32)
            if not tf.equal(tf.size(pidx), 0):
                self.matchpatcount.assign(tf.size(pidx))
                ppreds = tf.gather(patbinpreds, indices=pidx, axis=0)
                ppreds.set_shape([None] + patbinpreds.shape[1:])
                patpatpreds = self.categorizer(ppreds)
                patpredsarray = patpredsarray.scatter(pidx, patpatpreds)

            # Redo base prediction for pat-matching inputs with low conf
            bidx = tf.cast(tf.reshape(tf.where(tf.squeeze(patbinpreds) < 0.5), shape=[-1]), dtype=tf.int32)
            if not tf.equal(tf.size(bidx), 0):
                self.matchbasecount.assign(tf.size(bidx))
                bmatches = tf.gather(fmatches, indices=bidx, axis=0)
                bmatches.set_shape([None] + fmatches.shape[1:])
                patbasepreds = self.base_pred(bmatches)
                patpredsarray = patpredsarray.scatter(bidx, patbasepreds)

            patpreds = patpredsarray.stack()
            mergedpreds = mergedpreds.scatter(midx, patpreds)

        nonmatches = tf.math.logical_not(matches)
        nidx = tf.cast(tf.reshape(tf.where(nonmatches), shape=[-1]), dtype=tf.int32)
        if not tf.equal(tf.size(nidx), 0):
            self.nonmatchcount.assign(tf.size(nidx))
            fnonmatches = tf.gather(feats[0], indices=nidx, axis=0)
            fnonmatches.set_shape([None] + feats[0].shape[1:])   # Needed for SymbolicTensor
            basepreds = self.base_pred(fnonmatches)
            mergedpreds = mergedpreds.scatter(nidx, basepreds)

        outs = mergedpreds.stack()
        return outs


class PatternTrainer(layers.Layer):
    def __init__(self, feat_extract, pat_pred, pat_index, **kwargs):
        super().__init__(**kwargs)
        self.feat_extract = feat_extract
        self.feat_extract.trainable = False
        self.pat_pred = pat_pred
        self.pat_index = pat_index

    def call(self, inputs):
        feats = self.feat_extract(inputs)
        selectedfeats = tf.gather(feats[0], indices=self.pat_index, axis=3)
        preds = self.pat_pred(selectedfeats)
        return preds


def build_pattern_model(pattern, base_model, trans_model):
    pattern = list(pattern)
    pattern.sort()

    # Feature extraction
    patlayername = mlutil.parse_feature_ref(pattern[0])[0]
    featextract = build_feature_extraction(base_model, [patlayername], False)

    # Base prediction
    patlayer = base_model.get_layer(patlayername)
    pi = base_model.layers.index(patlayer) + 1
    basepred = Model(inputs=base_model.layers[pi].input, outputs=base_model.output, name='base_pred')

    # Pattern matcher
    blayer = trans_model.get_layer('pat_binarize')
    matcher = MatchLayer(blayer, pattern)

    s = featextract.output_shape[0]
    inputs = keras.Input(shape=(s[1], s[2], len(pattern)))
    x = layers.MaxPooling2D((2, 2))(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    dense1 = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    #dense3 = BinaryToCategorical(0, 3)(dense1)
    pattrain = Model(inputs=inputs, outputs=dense1, name='pat_train')
    #patpred = Model(inputs=inputs, outputs=dense3, name='pat_pred')

    inputs = keras.Input(shape=featextract.input_shape[1:])
    x = PatternTrainer(featextract, pattrain, matcher.pat_index)(inputs)
    pat_trainer = Model(inputs=inputs, outputs=x)
    pat_trainer.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    inputs = keras.Input(base_model.input_shape[1:])
    x = PatternBranch(featextract, matcher, basepred, pattrain)(inputs)
    pmodel = Model(inputs=inputs, outputs=x)
    pmodel.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return pmodel, pat_trainer


def pat_evaluate(model, ds):
    numbatches = tf.data.experimental.cardinality(ds).numpy()
    p = Progbar(numbatches)
    matchcount = 0
    matchpatcount = 0
    matchbasecount = 0
    nonmatchcount = 0
    acc = keras.metrics.CategoricalAccuracy()
    for step, (x_batch, y_batch) in enumerate(ds):
        #y_pred = model.predict(x_batch)
        y_pred = model(x_batch)
        matchcount += model.layers[-1].matchcount.numpy()
        matchpatcount += model.layers[-1].matchpatcount.numpy()
        matchbasecount += model.layers[-1].matchbasecount.numpy()
        nonmatchcount += model.layers[-1].nonmatchcount.numpy()
        acc.update_state(y_batch, y_pred)
        p.update(step + 1)
    a = acc.result()
    print(a)