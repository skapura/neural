import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional, Model
import pandas as pd
import numpy as np
import mlutil
import data
import patterns as pats
import const


@tf.keras.utils.register_keras_serializable()
class PatternBranch(keras.Layer):
    def __init__(self, pattern_set=None, **kwargs):
        super().__init__(**kwargs)
        self.pattern_models = None
        self.pattern_set = pattern_set

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        base_config = super().get_config()
        config = {'pattern_set': keras.saving.serialize_keras_object(self.pattern_set)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        pat_config = config.pop('pattern_set')
        pattern_set = keras.saving.deserialize_keras_object(pat_config)
        return cls(pattern_set, **config)

    def call(self, inputs):
        return inputs


def build_branch_model(base_model, branch_on_layer, pattern_set):
    base_model.trainable = False
    player = base_model.get_layer(branch_on_layer)

    patfeatmodel = Functional(inputs=base_model.input, outputs=player.output)

    x = PatternBranch(pattern_set)(player.output)
    x = layers.Conv2D(128, (3, 3), name='pat_conv')(player.output)
    x = layers.Activation("relu", name='pat_activation')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    model = Functional(base_model.input, x)
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(1)


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    #x = PatternBranch()(x)
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


class PatternLayer(layers.Layer):
    def __init__(self, base_model, pattern, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.pattern = pattern

        # Build feature extraction model for pattern detection
        player = base_model.get_layer('activation')
        self.pat_feat_extract = Model(inputs=base_model.inputs, outputs=[player.output])

        # Build remainder of base model after pattern detection
        nextlayer = base_model.layers.index(player) + 1
        self.base_model_prediction = Model(inputs=base_model.layers[nextlayer].input, outputs=base_model.output)

        # Build pattern branch
        o = list(self.pat_feat_extract.output_shape)
        o[-1] = len(pattern)
        inputs = keras.Input(shape=tuple(o[1:]))
        x = layers.Conv2D(16, (3, 3))(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
        self.pat_model = Functional(inputs, x)

        print('init')

    def build(self, input_shape):
        print('build')
        self.built = True

    def call(self, inputs, training=False):
        print('call')
        feats = self.pat_feat_extract(inputs)
        predictions = self.base_model_prediction(feats)
        return predictions


class PatternModel(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        print(1)



def run():
    model = models.load_model('largeimage16.keras', compile=True)
    x = PatternLayer(model, ['activation-0', 'activation-1'])(model.input)
    pmodel = Model(model.input, x)
    #pmodel = build_model()
    #pmodel.set_weights(model.get_weights())
    # pmodel.save('session/patmodel.keras')
    # loaded = models.load_model('session/patmodel.keras')

    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    # franges = get_ranges(trans, zeromin=True)
    scaled = data.scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    # col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation-' in c]
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]

    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems = pats.unique_elements(cpats)

    v = list(elems.keys())
    build_branch_model(model, 'activation', [{'filters': v, 'class': 0}])
    #pmodel = models.load_model('session/patmodel.keras')
    #player = pmodel.get_layer('pattern_branch')
    #player.build_branch([{'filters': v, 'class': 0}])
    #pmodel.save('session/patmodel.keras')
    # loaded = models.load_model('session/patmodel.keras')
    print(1)
