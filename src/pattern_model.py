import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Model
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
        self.matchcount = 0
        self.nonmatchcount = 0

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return [None, 3]

    def call(self, inputs):
        feats = self.feat_extract(inputs)

        # Separate inputs matching pattern
        matches = self.pat_matcher(feats[1])
        midx = tf.squeeze(tf.where(matches))
        midx = tf.cast(midx, dtype=tf.int32)
        fmatches = tf.gather(feats[0], indices=midx, axis=0)
        fmatches.set_shape([None] + feats[0].shape[1:])
        #self.matchcount += len(fmatches)
        selectedfeats = tf.gather(fmatches, indices=self.pat_matcher.pat_index, axis=3)
        patpreds = self.pat_pred(selectedfeats)

        nonmatches = tf.math.logical_not(matches)
        nidx = tf.squeeze(tf.where(nonmatches))
        nidx = tf.cast(nidx, dtype=tf.int32)
        fnonmatches = tf.gather(feats[0], indices=nidx, axis=0)
        fnonmatches.set_shape([None] + feats[0].shape[1:])   # Needed for SymbolicTensor
        #self.nonmatchcount += len(fnonmatches)
        basepreds = self.base_pred(fnonmatches)

        mergedpreds = tf.TensorArray(inputs.dtype, dynamic_size=True, size=0)
        mergedpreds = mergedpreds.scatter(midx, patpreds)
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
    dense3 = BinaryToCategorical(0, 3)(dense1)
    pattrain = Model(inputs=inputs, outputs=dense1, name='pat_train')
    patpred = Model(inputs=inputs, outputs=dense3, name='pat_pred')

    inputs = keras.Input(shape=featextract.input_shape[1:])
    x = PatternTrainer(featextract, pattrain, matcher.pat_index)(inputs)
    pat_trainer = Model(inputs=inputs, outputs=x)
    pat_trainer.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    inputs = keras.Input(base_model.input_shape[1:])
    x = PatternBranch(featextract, matcher, basepred, patpred)(inputs)
    pmodel = Model(inputs=inputs, outputs=x)
    pmodel.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return pmodel, pat_trainer
