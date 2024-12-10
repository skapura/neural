import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Model
from keras.src.utils import Progbar
import pandas as pd
import numpy as np
import mlutil


# Transform sets of feature maps into transactions
@tf.keras.utils.register_keras_serializable()
class TransactionLayer(layers.Layer):

    def transform_instance(self, x):
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.map_fn(tf.reduce_max, x)
        return x

    @tf.function
    def call(self, inputs, training=None):
        outputs = tf.map_fn(self.transform_instance, inputs)
        return outputs


# Transform transaction into binary transaction
@tf.keras.utils.register_keras_serializable()
class BinarizeLayer(layers.Layer):
    def __init__(self, layer_names=None, feature_names=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_names = layer_names
        self.feature_names = feature_names
        self.medians = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_names': self.layer_names,
            'feature_names': self.feature_names
        })
        return config

    def layer_medians(self, layer_name):
        startidx = self.feature_names.index(next((x for x in self.feature_names if x.startswith(layer_name)), None))
        endidx = self.feature_names.index(next((x for x in reversed(self.feature_names) if x.startswith(layer_name)), None))
        return self.medians[startidx:endidx+1]

    def build(self, input_shape):
        super().build(input_shape)
        self.medians = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='medians')

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
    def binarize_median(self, x):
        return tf.cond(tf.math.logical_and(tf.math.greater(x[1], 0.0), tf.math.greater_equal(x[0], x[1])),
                       lambda: tf.constant(True), lambda: tf.constant(False))

    @tf.function
    def binarizer(self, x):
        c = tf.stack([x, self.medians], axis=1)
        r = tf.map_fn(self.binarize_median, c, fn_output_signature=tf.bool)
        return r

    @tf.function
    def call(self, inputs, training=None):
        binned = tf.map_fn(self.binarizer, inputs, fn_output_signature=tf.bool)
        return binned


# Build feature extraction model
# if transaction_only=True, return feat_extact + base model prediction
# if transaction_only=False, feat_extract includes raw layer output as well as extracted features
def build_feature_extraction(base_model, output_layers=None, transaction_only=True):
    base_model.trainable = False
    layermodel = mlutil.make_output_nodes(base_model, output_layers)
    if output_layers is None:
        output_layers = list(layermodel.output_names)

    inputs = keras.Input(base_model.input_shape[1:])
    layermodel.name = 'feat_extract'
    xb = layermodel(inputs)
    x = [TransactionLayer(name='trans_' + lname)(xo) for xo, lname in zip(xb[:-1], output_layers[:-1])]
    x = layers.Concatenate(name='transaction')(x)
    if transaction_only:
        model = Model(inputs=inputs, outputs=x, name='feats_to_trans')
        return model, xb[-1]
    else:
        outs = [layers.Identity()(xe) for xe in xb[:-1]]
        outs.append(x)
        model = Model(inputs=inputs, outputs=outs, name='feats_to_trans')
        return model


# Build transaction model
# feat_extract -> binarizer
def build_transaction_model(base_model, trainds, output_layers=None):
    featextract, baseoutput = build_feature_extraction(base_model, output_layers)
    if output_layers is None:
        output_layers = [l.name[6:] for l in featextract.layers if isinstance(l, TransactionLayer)]
        output_layers.append('prediction')

    # Generate list of feature names
    featnames = []
    for i in range(len(output_layers) - 1):
        layername = output_layers[i]
        nodecount = featextract.layers[2 + i].input.shape[-1]
        for n in range(nodecount):
            featnames.append(layername + '-' + str(n))

    # Append binarizer layer to feature extraction
    x = BinarizeLayer(name='pat_binarize', layer_names=output_layers, feature_names=featnames)(featextract.output)
    xd = layers.Identity()(baseoutput)
    model = Model(inputs=featextract.inputs, outputs=[x, xd])
    medians = fit_medians(featextract, trainds)
    model.get_layer('pat_binarize').set_weights([medians])

    return model


@tf.function
def calc_median(x):
    m = x.get_shape()[0] // 2
    med = tf.reduce_min(tf.nn.top_k(x, m, sorted=False).values)
    return med


@tf.function
def calc_medians(x):
    vals = tf.transpose(x)
    medians = tf.map_fn(calc_median, vals)
    return medians


def fit_medians(model, ds):
    bds = ds.repeat(1)
    numbatches = tf.data.experimental.cardinality(ds).numpy()
    p = Progbar(numbatches)
    r = []
    for i, (x, y) in enumerate(bds):
        outs = model(x)
        r.append(outs)
        p.update(i)
    c = tf.concat(r, axis=0)

    medians = calc_medians(c)
    return medians


# Convert set of transactions to dataframe
# if ds=None, only include transactions
# if ds!=None, append predictions and image path
def transactions_to_dataframe(model, trans, ds=None):
    df = pd.DataFrame(trans[0], columns=model.get_layer('pat_binarize').feature_names)
    df.index.name = 'index'

    if ds is not None:
        bds = ds.repeat(1)
        labels = []
        for i, batch in enumerate(bds):
            labels += [np.argmax(x) for x in batch[1]]
        df['predicted'] = [np.argmax(x) for x in trans[1]]
        df['label'] = labels
        df['path'] = ds.file_paths

    return df
