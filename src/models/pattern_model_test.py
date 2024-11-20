import tensorflow as tf
from keras import models, layers
import keras
import pandas as pd
from keras.src.models import Functional
import src.models.trans_model as tm
import src.models.pattern_model as pm
import src.data as data

def build_base_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3))(x)
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


def MatchLayer_test():
    #tf.config.run_functions_eagerly(True)
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)
    #basemodel = build_base_model()
    #basemodel.fit(trainds, epochs=10)
    #basemodel.save('session/base.keras')
    basemodel = models.load_model('session/base.keras')
    outputlayers = ['activation']
    #tmodel = tm.build_transaction_model(basemodel, trainds, outputlayers)
    #tmodel.save('session/tmodel.keras')
    tmodel = models.load_model('session/tmodel.keras')
    bdf = pd.read_csv('session/bdf.csv', index_col='index')
    patterns = [
        ['activation-0', 'activation-3', 'activation-4'],
        ['activation-6', 'activation-7', 'activation-8']
    ]

    pmodel = pm.build_pattern_model(patterns, basemodel, tmodel)

    #pds = pm.match_dataset(pmodel, trainds, binary_target='church')
    #data.save_subset(pds, 'session/pds.pickle')
    pds = data.load_subset('session/pds.pickle')
    pmodel.train_model.fit(pds, epochs=10)
    #pm.save_pattern_model(pmodel, 'session/pmodel.zip')
    pmodel2 = pm.load_pattern_model('session/pmodel.zip')
    pmodel2.train_model.fit(pds, epochs=5)
    pw = pmodel.get_weights()
    pw2 = pmodel2.get_weights()

    r = pmodel.train_model.evaluate(pds, return_dict=True)
    r2 = pmodel2.train_model.evaluate(pds, return_dict=True)
    print(r)
    print(r2)
    print(1)

def run():
    MatchLayer_test()
    print(1)
