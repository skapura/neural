import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Model
from keras.src.utils import Progbar
import mlutil


# Transform sets of feature maps into transactions
@tf.keras.utils.register_keras_serializable()
class TransactionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform_instance(self, x):
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.map_fn(tf.reduce_max, x)
        return x

    def call(self, inputs, training=None):
        outputs = tf.map_fn(self.transform_instance, inputs)
        return outputs


# Transform transaction into binary transaction
@tf.keras.utils.register_keras_serializable()
class BinarizeLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.medians = None

    def build(self, input_shape):
        super().build(input_shape)
        self.medians = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='medians')

    def compute_output_shape(self, input_shape):
        return input_shape

    def binarize_median(self, x):
        return tf.cond(tf.math.logical_and(tf.math.greater(x[1], 0.0), tf.math.greater_equal(x[0], x[1])),
                lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    def binarizer(self, x):
        c = tf.stack([x, self.medians], axis=1)
        r = tf.map_fn(self.binarize_median, c)
        return r

    def call(self, inputs, training=None):
        binned = tf.map_fn(self.binarizer, inputs)
        return binned


@tf.function
def median_caller(model, batch):
    outs = model(batch, training=True)
    return outs

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


def build_feature_extraction(base_model, output_layers=None):
    base_model.trainable = False
    layermodel = mlutil.make_output_nodes(base_model, output_layers)

    inputs = keras.Input(base_model.input_shape[1:])
    xb = layermodel(inputs)
    x = [TransactionLayer(name='trans_' + lname)(xo) for xo, lname in zip(xb[:-1], output_layers[:-1])]
    x = layers.Concatenate()(x)
    model = Model(inputs=inputs, outputs=x)
    return model, xb

def build_transaction_model(base_model, trainds, output_layers=None):
    featextract, baseoutput = build_feature_extraction(base_model, output_layers)
    medians = fit_medians(featextract, trainds)

    x = BinarizeLayer(name='pat_binarize')(featextract.output)
    xd = layers.Identity()(baseoutput[-1])
    model = Model(inputs=featextract.inputs, outputs=[x, xd])
    model.get_layer('pat_binarize').set_weights([medians])

    return model


def fit_medians(model, ds):
    bds = ds.repeat(1)
    numbatches = tf.data.experimental.cardinality(ds).numpy()
    p = Progbar(numbatches)
    r = []
    i = 0
    for x, y in bds:
        outs = median_caller(model, x)
        r.append(outs)
        i += 1
        p.update(i)
    c = tf.concat(r, axis=0)

    medians = calc_medians(c)
    return medians
    #model.get_layer('pat_binarize').set_weights([medians])
