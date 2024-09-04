import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional, Model
import pandas as pd
import os
import numpy as np
import joblib
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


@tf.keras.utils.register_keras_serializable()
class PatternLayer(layers.Layer):
    def __init__(self, base_model, pattern, pattern_class, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_model.trainable = False
        self.pattern = pattern
        self.pattern_class = pattern_class
        self.scaler = None

        # Build feature extraction model for pattern detection
        player = base_model.get_layer('activation')
        self.pat_feat_extract = Model(inputs=base_model.inputs, outputs=[player.output])

        # Build remainder of base model after pattern detection
        nextlayer = base_model.layers.index(player) + 1
        self.base_model_prediction = Model(inputs=base_model.layers[nextlayer].input, outputs=base_model.output)

        # Pre-pattern branch
        pidx = base_model.layers.index(base_model.get_layer('activation'))
        player = base_model.layers[pidx - 2]    # layer just before pattern branch
        self.pre_pat_model = Model(inputs=base_model.inputs, outputs=player.output)

        # Pattern selection layer
        cfg = base_model.layers[pidx - 1].get_config()    # full conv2d layer config
        cfg['filters'] = len(self.pattern)
        cfg.pop('name', None)
        patlayer = layers.Conv2D(**cfg)
        x = patlayer(player.output)
        filteridx = [int(e.split('-')[1]) for e in self.pattern]
        w = mlutil.slice_weights(base_model, 'activation', filteridx)
        patlayer.set_weights(w)
        cfg = base_model.layers[pidx].get_config()
        cfg.pop('name', None)
        x = layers.Activation(**cfg)(x)

        # Pattern branch after pattern selection
        x = layers.Conv2D(16, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
        self.pat_model = Model(inputs=self.pre_pat_model.inputs, outputs=x)
        self.pat_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])

        # Build pattern branch training model
        # self.pat_feat_extract.trainable = False
        # x = layers.Conv2D(16, (3, 3), name='pat_conv2d')(self.pat_feat_extract.layers[-1].output)
        # x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
        # self.pat_model_train = Model(inputs=self.pat_feat_extract.inputs, outputs=x)

        # Build pattern branch inference model
        #o = list(self.pat_feat_extract.output_shape)
        #o[-1] = len(pattern)
        #inputs = keras.Input(shape=tuple(o[1:]))
        #x = layers.Conv2D(16, (3, 3), name='pat_conv2d')(inputs)
        #x = layers.GlobalAveragePooling2D()(x)
        #x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
        #self.pat_model = Functional(inputs, x)

        print(1)

    #def save_assets(self, inner_path):
    #    joblib.dump(self.scaler, os.path.join(inner_path, 'scaler.save'))

    #def load_assets(self, inner_path):
    #    self.scaler = joblib.load(os.path.join(inner_path, 'scaler.save'))


    #def get_config(self):
    #    base_config = super().get_config()
    #    config = {'pattern_set': keras.saving.serialize_keras_object(self.pattern_set)}
    #    return {**base_config, **config}

    def call(self, inputs, training=False):
        # if training:    # Only training pattern branches
        #     feats = self.pat_feat_extract(inputs)
        #     em = [int(e.split('-')[1]) for e in self.pattern]
        #     selfeats = feats.numpy()[..., em]
        #     predictions = self.pat_model(selfeats)
        # else:
        #     #feats = self.pat_feat_extract(inputs)
        #     #predictions = self.base_model_prediction(feats)
        #     feats = self.pat_feat_extract(inputs)
        #     em = [int(e.split('-')[1]) for e in self.pattern]
        #     selfeats = feats.numpy()[..., em]
        #     predictions = self.pat_model(selfeats)
        #if len(inputs) == 1 and isinstance(inputs[0], tf.SymbolicTensor):
        feats = self.pat_feat_extract(inputs)
        predictions = self.base_model_prediction(feats)
        return predictions


class PatternModel(models.Model):
    def __init__(self, base_model, pattern, pattern_class, **kwargs):
        x = PatternLayer(base_model, pattern, pattern_class)(base_model.input)
        super().__init__(base_model.input, x, **kwargs)

    def fit_pattern(self, trainds, **kwargs):
        player = self.layers[-1]
        pattern = set(player.pattern)
        classname = trainds.class_names[player.pattern_class]

        #layermodel = mlutil.make_output_model(player.base_model)
        #trans = pats.model_to_transactions(layermodel, trainds, include_meta=True)
        #trans.to_csv('session/trans_feat.csv')
        trans = pd.read_csv('session/trans_feat_full.csv', index_col='index')
        scaled, player.scaler = data.scale(trans, output_range=(0, 1))
        return
        #joblib.dump(player.scaler, 'session/scaler.save')
        #scaler2 = joblib.load('session/scaler.save')
        bdf = pats.binarize(scaled, 0.5)

        matches, nonmatches = pats.matches(bdf, pattern)
        matchds = data.load_dataset_selection(trainds, trans.loc[matches]['path'].to_list())
        binds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        valbinds = None
        if 'validation_data' in kwargs:
            valds = kwargs['validation_data']

            #layermodel = mlutil.make_output_model(player.base_model)
            #trans = pats.model_to_transactions(layermodel, valds, include_meta=True)
            #trans.to_csv('session/vtrans_feat_full.csv')
            trans = pd.read_csv('session/vtrans_feat_full.csv', index_col='index')
            scaled, _ = data.scale(trans, output_range=(0, 1), scaler=scaler)
            bdf = pats.binarize(scaled, 0.5)

            matches, nonmatches = pats.matches(bdf, pattern)
            matchds = data.load_dataset_selection(valds, trans.loc[matches]['path'].to_list())
            valbinds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        player.pat_model.fit(binds, validation_data=valbinds, epochs=2)
        print(1)


def run():


    model = models.load_model('largeimage16.keras', compile=True)
    #x = PatternLayer(model, ['activation-0', 'activation-1'])(model.input)
    #pmodel = PatternModel(model.input, x)
    pmodel = PatternModel(model, ['activation-1', 'activation-2'], 1)
    pmodel.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'], run_eagerly=True)
    trainds, valds = data.load_dataset('images_large')
    pmodel.fit_pattern(trainds, validation_data=valds, epochs=3)
    pmodel.save('session/testpatmodel.keras')
    pmodel2 = models.load_model('session/testpatmodel.keras')

    #test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    results = model.evaluate(valds)
    presults = pmodel.evaluate(valds)
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
