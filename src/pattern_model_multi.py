import tensorflow as tf
from keras import layers, models
from keras.src.models import Model
import pandas as pd
import numpy as np
import joblib
import os
import data
import patterns as pats
import mlutil


@tf.keras.utils.register_keras_serializable()
class PatternLayerMulti(layers.Layer):

    def __init__(self, patterns, **kwargs):
        super().__init__(**kwargs)
        self.patterns = patterns
        # self.pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern]
        # self.pattern_class = pattern_class
        for p in self.patterns:
            p['pattern_feats'] = [int(mlutil.parse_feature_ref(e)[1]) for e in p['pattern']]
            p['layer_name'] = mlutil.parse_feature_ref(p['pattern'][0])[0]
            p['pat_model'] = None
            p['pat_feat_extract'] = None
        # self.layer_name = mlutil.parse_feature_ref(pattern[0])[0]
        self.build_output_shape = [None, 1]
        self.base_model = None
        #self.pat_model = None
        #self.pat_feat_extract = None
        #self.base_model_prediction = None
        #self.pre_pat_model = None
        #self.pat_model_prediction = None
        self.scaler = None

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def build_branch(self, base_model):
        self.base_model = base_model
        self.base_model.trainable = False

        for p in self.patterns:
            pidx = self.base_model.layers.index(self.base_model.get_layer(p['layer_name']))
            beforepat = self.base_model.layers[pidx - 2]

            # Pattern-filter conv2d
            cfg = base_model.layers[pidx - 1].get_config()
            cfg['filters'] = len(p['pattern'])
            cfg.pop('name', None)
            patlayer = layers.Conv2D(**cfg)
            x = patlayer(beforepat.output)

            # Pattern-filter activation
            filteridx = [int(e.split('-')[1]) for e in p['pattern']]
            w = mlutil.slice_weights(base_model, p['layer_name'], filteridx)
            patlayer.set_weights(w)
            cfg = base_model.layers[pidx].get_config()
            cfg.pop('name', None)
            x = layers.Activation(**cfg)(x)

            # Pattern branch after pattern filter
            x = layers.Conv2D(16, (3, 3), name='pat_hook')(x)
            x = layers.Activation('relu')(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1, name='prediction', activation='sigmoid')(x)

            # Build pattern model
            p['pat_model'] = Model(inputs=self.base_model.inputs, outputs=x)
            p['pat_model'].compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                   metrics=[tf.keras.metrics.BinaryAccuracy()])

    def build_model_segments(self):


@tf.keras.utils.register_keras_serializable()
class PatternModelMulti(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pat_layer = self.layers[-1]

    @staticmethod
    def make(base_model, patterns, scaler=None):
        player = PatternLayerMulti(patterns)
        player.scaler = scaler
        player.build_branch(base_model)
        x = player(base_model.input)
        pmodel = PatternModelMulti(inputs=base_model.input, outputs=x)
        return pmodel
