import tensorflow as tf
from keras import models, layers
import keras
from keras.src.models import Functional
import pandas as pd
import models.trans_model as tm
import models.pattern_model as pm
import data as data
import patterns as pats
from azure import AzureSession


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


def run_model():
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)

    # Load support files
    basemodel = models.load_model('session/base.keras')
    tmodel = models.load_model('session/tmodel.keras')
    bdf = pd.read_csv('session/bdf.csv', index_col='index')

    # Mine patterns
    minsup = 0.05
    minsupratio = 1.1
    sel, notsel = data.filter_transactions(bdf, 'activation_2-', 2)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    patterns = [list(p['pattern']) for p in cpats[:2]]

    pmodel = pm.build_pattern_model(patterns, basemodel, tmodel)

    # Matching data set
    pds = pm.match_dataset(pmodel, trainds, binary_target='golf_ball')
    pds_b = pm.match_dataset(pmodel, trainds)
    #data.save_subset(pds, 'session/pds.pickle')
    #pds = data.load_subset('session/pds.pickle')

    # Pattern model
    pmodel.train_model.fit(pds, epochs=10)
    pm.save_pattern_model(pmodel, 'session/pmodel.zip')
    r = pmodel.train_model.evaluate(pds, return_dict=True)
    rb = basemodel.evaluate(pds_b, return_dict=True)
    print(r)
    print(rb)


def make_features():
    print(1)

def eval():
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)
    pmodel = pm.load_pattern_model('session/pmodel.zip')

    for i, (xbatch, ybatch) in enumerate(trainds):
        y = pmodel(xbatch)
        #yb = pmodel.base_model(xbatch)
        print(i)

    print(1)


def run():
    #tf.config.run_functions_eagerly(True)
    #trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)

    # Base model
    #basemodel = build_base_model()
    #basemodel.fit(trainds, epochs=10)
    #basemodel.save('session/base.keras')
    #basemodel = models.load_model('session/base.keras')

    # Transactions
    outputlayers = ['activation_2']
    #tmodel = tm.build_transaction_model(basemodel, trainds, outputlayers)
    #tmodel.save('session/tmodel.keras')
    #tmodel = models.load_model('session/tmodel.keras')
    #trans = tmodel.predict(trainds)
    #bdf = tm.transactions_to_dataframe(tmodel, trans, trainds)
    #bdf.to_csv('session/bdf.csv')
    #bdf = pd.read_csv('session/bdf.csv', index_col='index')

    eval()


    #run_model()

    session = AzureSession()
    session.open(working_dir='neural')
    session.put('session/base.keras', 'neural/session')
    session.put('session/tmodel.keras', 'neural/session')
    session.put('session/bdf.csv', 'neural/session')
    session.upload_function(run_model)
    session.execute('python src/remote_spec.py')
    session.get('neural/session/pmodel.zip', 'session')
    session.close()

    run_model()

    minsup = 0.5
    minsupratio = 1.1
    sel, notsel = data.filter_transactions(bdf, 'activation-', 0)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    patterns = [list(p['pattern']) for p in cpats[:10]]

    pmodel = pm.build_pattern_model(patterns, basemodel, tmodel)

    pds = pm.match_dataset(pmodel, trainds, binary_target='church')
    data.save_subset(pds, 'session/pds.pickle')
    pds = data.load_subset('session/pds.pickle')
    pmodel.train_model.fit(pds, epochs=10)
    #pm.save_pattern_model(pmodel, 'session/pmodel.zip')
    #pmodel2 = pm.load_pattern_model('session/pmodel.zip')
    #pmodel2.train_model.fit(pds, epochs=5)
    #pw = pmodel.get_weights()
    #pw2 = pmodel2.get_weights()

    r = pmodel.train_model.evaluate(pds, return_dict=True)
    #r2 = pmodel2.train_model.evaluate(pds, return_dict=True)
    print(r)
    #print(r2)
    print(1)
