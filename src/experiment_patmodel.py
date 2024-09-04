import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional, Model
import pandas as pd
import os
import numpy as np
import mlutil
import data
import patterns as pats
import const


class PatternSelector(layers.Layer):
    def __init__(self, pattern, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.pattern_features = [int(e.split('-')[1]) for e in self.pattern]

    #def compute_output_shape(self, *args, **kwargs):

    def build(self, input_shape):
        print(1)

    def call(self, inputs):
        #s = inputs.shape
        #inputs.shape[-1] = 2
        a = list(inputs.shape)
        a[-1] = len(self.pattern)
        #o = tf.ones(inputs.shape)
        inputs.set_shape((None, 254, 254, 2))
        #selfeats = inputs.numpy()[..., self.pattern_features]
        return inputs


def run():
    model = models.load_model('largeimage16.keras', compile=True)
    model.summary()
    a = model.layers[2].get_config()
    b = layers.Activation(**a)
    player = model.get_layer('activation')
    pat_feat_extract = Model(inputs=model.inputs, outputs=[player.output])
    pat_feat_extract.trainable = False
    x = PatternSelector(['activation-1', 'activation-2'])(player.output)
    x = layers.Conv2D(16, (3, 3), name='pat_conv2d')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    pat_model = Model(inputs=pat_feat_extract.inputs, outputs=x)
    self.pat_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                       metrics=[tf.keras.metrics.BinaryAccuracy()])
