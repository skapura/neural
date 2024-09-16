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
class PatternLayer(layers.Layer):

    def __init__(self, pattern, pattern_class, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern]
        self.pattern_class = pattern_class
        self.layer_name = mlutil.parse_feature_ref(pattern[0])[0]
        self.build_output_shape = [None, 1]
        self.base_model = None
        self.pat_model = None
        self.pat_feat_extract = None
        self.base_model_prediction = None
        self.pre_pat_model = None
        self.pat_model_prediction = None
        self.scaler = None

    def get_build_config(self):
        build_config = {"output_shape": self.output.shape}
        return build_config

    def build_from_config(self, config):
        self.build_output_shape = config['output_shape']

    def save_assets(self, inner_path):
        joblib.dump(self.scaler, os.path.join(inner_path, 'scaler.save'))
        self.base_model.trainable = True
        self.base_model.save(os.path.join(inner_path, 'base_model.keras'))
        self.pat_model.save(os.path.join(inner_path, 'pat_model.keras'))

    def load_assets(self, inner_path):
        self.scaler = joblib.load(os.path.join(inner_path, 'scaler.save'))
        self.base_model = models.load_model(os.path.join(inner_path, 'base_model.keras'), compile=True)
        self.pat_model = models.load_model(os.path.join(inner_path, 'pat_model.keras'), compile=True)
        self.build_model_segments()

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def build_branch(self, base_model):
        self.base_model = base_model
        self.base_model.trainable = False

        pidx = self.base_model.layers.index(self.base_model.get_layer(self.layer_name))
        beforepat = self.base_model.layers[pidx - 2]

        # Pattern-filter conv2d
        cfg = base_model.layers[pidx - 1].get_config()
        cfg['filters'] = len(self.pattern)
        cfg.pop('name', None)
        patlayer = layers.Conv2D(**cfg)
        x = patlayer(beforepat.output)

        # Pattern-filter activation
        filteridx = [int(e.split('-')[1]) for e in self.pattern]
        w = mlutil.slice_weights(base_model, self.layer_name, filteridx)
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
        self.pat_model = Model(inputs=self.base_model.inputs, outputs=x)
        self.pat_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                               metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.build_model_segments()

    def build_model_segments(self):

        # Build feature extraction model for pattern detection
        player = self.base_model.get_layer(self.layer_name)
        self.pat_feat_extract = Model(inputs=self.base_model.inputs, outputs=player.output)

        # Build remainder of base model after pattern detection
        nextlayer = self.base_model.layers.index(player) + 1
        self.base_model_prediction = Model(inputs=self.base_model.layers[nextlayer].input, outputs=self.base_model.output)

        # Pattern branch
        hooklayer = self.pat_model.get_layer('pat_hook')
        self.pat_model_prediction = Model(inputs=hooklayer.input, outputs=self.pat_model.output)

    def fit(self, ds, **kwargs):
        pattern = set(self.pattern)
        classname = ds.class_names[self.pattern_class]

        # Filter training data
        transpath = kwargs['trans_path'] if 'trans_path' in kwargs else None
        bdf, s = pats.preprocess(self.base_model, ds, transpath, self.scaler)
        if self.scaler is None:
            self.scaler = s
        matches, nonmatches = pats.matches(bdf, pattern)
        matchds = data.load_dataset_selection(ds, bdf.loc[matches]['path'].to_list())
        binds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Filter validation data
        valbinds = None
        if 'validation_data' in kwargs:
            valds = kwargs['validation_data']
            valpath = kwargs['val_path'] if 'val_path' in kwargs else None
            bdf, _ = pats.preprocess(self.base_model, valds, valpath, self.scaler)
            matches, nonmatches = pats.matches(bdf, pattern)
            matchds = data.load_dataset_selection(valds, bdf.loc[matches]['path'].to_list())
            valbinds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Train pattern branch
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1
        self.pat_model.fit(binds, validation_data=valbinds, epochs=epochs)

    def call(self, inputs):
        feats = self.pat_feat_extract(inputs)
        predictions = self.base_model_prediction(feats)
        return predictions

        trans = pats.features_to_transactions(feats, self.layer_name)

        firstindex = list(self.scaler.feature_names_in_).index(self.layer_name + '-0')
        lastindex = list(self.scaler.feature_names_in_).index(self.layer_name + '-' + str(feats.shape[-1] - 1)) + 1
        self.scaler.feature_names_in_ = self.scaler.feature_names_in_[firstindex:lastindex]
        self.scaler.min_ = self.scaler.min_[firstindex:lastindex]
        self.scaler.scale_ = self.scaler.scale_[firstindex:lastindex]
        self.scaler.data_max_ = self.scaler.data_max_[firstindex:lastindex]
        self.scaler.data_min_ = self.scaler.data_min_[firstindex:lastindex]
        self.scaler.data_range_ = self.scaler.data_range_[firstindex:lastindex]
        self.scaler.n_features_in_ = len(self.scaler.feature_names_in_)
        scaled, _ = data.scale(trans, output_range=(0, 1), scaler=self.scaler, exclude=[])
        bdf = pats.binarize(scaled, 0.5, exclude=[])
        matches, nonmatches = pats.matches(bdf, set(self.pattern))

        patpreds = []
        basepreds = []
        for i, finput in enumerate(feats):
            if i in matches:
                selfeats = finput.numpy()[..., self.pattern_feats]
                branchpred = self.pat_model_prediction(np.asarray([selfeats]))
                patpreds.append(branchpred)
            else:
                basepred = self.base_model_prediction(np.asarray([finput]))
                basepreds.append(basepred)

        patpred = self.pat_model(inputs)

        selfeats = feats.numpy()[..., self.pattern_feats]
        branchpred = self.pat_model_prediction(selfeats)

        predictions = self.base_model_prediction(feats)
        return predictions


@tf.keras.utils.register_keras_serializable()
class PatternModel(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pat_layer = self.layers[-1]

    @staticmethod
    def make(base_model, pattern, pattern_class, scaler=None):
        patlist = list(pattern)
        patlist.sort()
        player = PatternLayer(patlist, pattern_class)
        player.scaler = scaler
        player.build_branch(base_model)
        x = player(base_model.input)
        pmodel = PatternModel(inputs=base_model.input, outputs=x)
        return pmodel
        
    def fit(self, ds, **kwargs):
        self.pat_layer.fit(ds, **kwargs)

    def evaluate(self, ds, trans_path=None):
        bdf, _ = pats.preprocess(self.pat_layer.base_model, ds, trans_path, self.pat_layer.scaler)
        patds, baseds = pats.match_dataset(ds, bdf, self.pat_layer.pattern, self.pat_layer.pattern_class)
        p = self.pat_layer.pat_model.evaluate(patds, return_dict=True)
        b = self.pat_layer.base_model.evaluate(baseds, return_dict=True)
        return {'base': b, 'pattern': p}

