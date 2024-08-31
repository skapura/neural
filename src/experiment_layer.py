import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
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
        self.pattern_set = pattern_set
        print('init pattern')

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


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = PatternBranch()(x)
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


def run():
    model = models.load_model('largeimage16.keras', compile=True)
    pmodel = build_model()
    pmodel.set_weights(model.get_weights())
    #pmodel.save('session/patmodel.keras')
    #loaded = models.load_model('session/patmodel.keras')

    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    #franges = get_ranges(trans, zeromin=True)
    scaled = data.scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    #col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation-' in c]
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]

    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems = pats.unique_elements(cpats)
    v = list(elems.keys())
    player = pmodel.get_layer('pattern_branch')
    player.pattern_set = [{'filters': v, 'class': 0}]
    pmodel.save('session/patmodel.keras')
    loaded = models.load_model('session/patmodel.keras')
    print(1)
