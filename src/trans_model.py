from keras import layers, models
import tensorflow as tf
import keras
from keras.src.models import Model
from patterns import feature_activation_max
import mlutil
import numpy as np


class TransactionLayer(layers.Layer):
    def __init__(self, feature_activation=feature_activation_max, **kwargs):
        super().__init__(**kwargs)
        self.feature_activation = feature_activation
        self.max_vals = np.empty(0)

    def transform_instance(self, x):
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.map_fn(tf.reduce_max, x)
        return x

    def call(self, inputs):
        outputs = tf.map_fn(self.transform_instance, inputs)
        return outputs

    @staticmethod
    def make_model(base_model, feature_activation=feature_activation_max):
        layermodel = mlutil.make_output_model(base_model)

        inputs = keras.Input(base_model.input_shape[1:])
        x = layermodel(inputs)
        x = [TransactionLayer(feature_activation)(xo) for xo in x[:-1]]
        x = layers.Concatenate()(x)
        x = layers.BatchNormalization()(x)
        x = MaxLayer()(x)
        tmodel = Model(inputs=inputs, outputs=x)
        return tmodel


class MaxLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_vals = np.empty(0)

    def call(self, inputs):
        if len(self.max_vals == 0):
            self.max_vals = np.zeros(inputs.shape[-1])
        return inputs
