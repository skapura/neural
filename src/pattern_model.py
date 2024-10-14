import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Model
from trans_model import build_feature_extraction#TransactionLayer, BinarizeLayer
import mlutil


class PatternMatch(layers.Layer):
    def __init__(self, trans_model, **kwargs):
        super().__init__(**kwargs)
        self.trans_model = trans_model

    def compute_output_shape(self, input_shape):
        return [None, 1]

    def call(self, inputs):
        outs = self.trans_model(inputs)
        print(1)

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


class PatternTest(layers.Layer):
    def __init__(self, feat_extract, base_pred, pat_matcher, **kwargs):
        super().__init__(**kwargs)
        self.feat_extract = feat_extract
        self.base_pred = base_pred
        self.pat_matcher = pat_matcher

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return [None, 3]

    def call(self, inputs):
        feats = self.feat_extract(inputs)

        # Separate inputs matching pattern
        matches = self.pat_matcher(feats[1])
        nonmatches = tf.math.logical_not(matches)
        idx = tf.squeeze(tf.where(matches))
        fmatches = tf.gather(feats[0], indices=idx, axis=0)
        fmatches.set_shape([None] + feats[0].shape[1:])
        idx = tf.squeeze(tf.where(nonmatches))
        fnonmatches = tf.gather(feats[0], indices=idx, axis=0)
        fnonmatches.set_shape([None] + feats[0].shape[1:])   # Needed for SymbolicTensor

        bpreds = self.base_pred(fmatches)

        outs = self.base_pred(feats[0])
        return outs


def build_pattern_model(pattern, base_model, trans_model):
    pattern = list(pattern)
    pattern.sort()
    patlayername = mlutil.parse_feature_ref(pattern[0])[0]
    featextract = build_feature_extraction(base_model, [patlayername], False)

    patlayer = base_model.get_layer(patlayername)
    pi = base_model.layers.index(patlayer) + 1
    basepred = Model(inputs=base_model.layers[pi].input, outputs=base_model.output)

    blayer = trans_model.get_layer('pat_binarize')
    matcher = MatchLayer(blayer, pattern)



    inputs = keras.Input(base_model.input_shape[1:])
    x = PatternTest(featextract, basepred, matcher)(inputs)
    pt = Model(inputs=inputs, outputs=x)
    pt.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return pt
