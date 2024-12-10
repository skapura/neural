import tensorflow as tf
from keras import models, layers
import keras
from keras.src.models import Functional
import pandas as pd
import numpy as np
import cv2
import models.trans_model as tm
import models.pattern_model as pm
import data as data
import patterns as pats
import mlutil
import plot
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


def build_base_model3():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
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


def build_base_model2():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def build_base_model1():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    #x = layers.Conv2D(32, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def build_base_model0():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(64, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Conv2D(32, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3))(x)
    #x = layers.Activation('relu')(x)
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


def train_base_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)
    model.fit(trainds, epochs=50)
    model.save('session/model.keras')


def make_features():
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)
    #base_model = build_base_model0()
    #base_model.fit(trainds, epochs=10)
    #base_model.save('session/base0b.keras')
    #base_model = models.load_model('session/base0.keras')
    #pmodel = pm.load_pattern_model('session/pmodel.zip')
    #bdf = pd.read_csv('session/bdf.csv', index_col='index')

    session = AzureSession()
    session.open(working_dir='neural')
    #session.put('session/base.keras', 'neural/session')
    #session.put('session/tmodel.keras', 'neural/session')
    #session.put('session/bdf.csv', 'neural/session')
    session.upload_function(train_base_model)
    session.execute('python src/remote_spec.py')
    session.get('neural/session/model.keras', 'session')
    session.close()

    bt = models.load_model('session/model.keras')
    rt = bt.evaluate(trainds, return_dict=True)
    print(rt)

    b3 = models.load_model('session/base3.keras')
    b2 = models.load_model('session/base2.keras')
    b1 = models.load_model('session/base1.keras')
    b0 = models.load_model('session/base0.keras')
    r3 = b3.evaluate(trainds, return_dict=True)
    r2 = b2.evaluate(trainds, return_dict=True)
    r1 = b1.evaluate(trainds, return_dict=True)
    r0 = b0.evaluate(trainds, return_dict=True)
    print(r3)
    print(r2)
    print(r1)
    print(r0)

    idx = -1
    cls = 2
    for i, (xbatch, ybatch) in enumerate(trainds):
        for j, (x, y) in enumerate(zip(xbatch, ybatch)):
            c = np.argmax(y)
            idx += 1
            if c != cls:
                continue
            print(idx)
            img = cv2.imread(trainds.file_paths[idx])
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig = cv2.resize(img, (256, 256))
            hmap, himg = mlutil.heatmap(np.asarray([x]), base_model, 'activation', cls)
            #cv2.imwrite('session/heat/heat.png', hmap)
            #cv2.imwrite('session/heat/heat2.png', himg)
            heatimage = plot.overlay_heatmap(orig, himg, alpha=0.5)
            #heatimage = cv2.cvtColor(heatimage, cv2.COLOR_RGB2BGR)
            cv2.imwrite('session/heat/' + str(idx) + '.png', heatimage)
            print(1)


    #for index, row in incorrect.iterrows():
    #    heatmap = heatmap(np.asarray([test_images[index]]), model, 'activation_2', row['predicted'])
    #    heatimage = overlayHeatmap(np.asarray([orig_test_images[index]]), heatmap)
    #    cv2.imwrite('heat/incorrect/' + str(index).zfill(5) + '.png', heatimage)
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
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=None)

    # Base model
    #basemodel = build_base_model()
    #basemodel.fit(trainds, epochs=10)
    #basemodel.save('session/base.keras')
    basemodel = models.load_model('session/base_good.keras')
    #tmodel = tm.build_transaction_model(basemodel, trainds, output_layers=None)
    #tmodel.save('session/tmodel_good.keras')
    tmodel = models.load_model('session/tmodel_good.keras')
    #trans = tmodel.predict(trainds)
    #bdf = tm.transactions_to_dataframe(tmodel, trans, trainds)
    #bdf.to_csv('session/bdf_good.csv')
    bdf = pd.read_csv('session/bdf_good.csv', index_col='index')

    #cpats = pats.find_patterns(bdf, 0, 'activation', min_sup=0.7)

    rdata = {'class': [], 'layer': [], 'counts': [], 'maxsup': [], 'maxratio': []}
    for c in [0, 1, 2]:
        print(c)
        for l in ['activation', 'activation_1', 'activation_2', 'activation_3']:
            print(l)
            rdata['class'].append(c)
            rdata['layer'].append(l)
            cpats = pats.find_patterns(bdf, c, l, min_sup=0.6)
            rdata['counts'].append(len(cpats))

            if len(cpats) > 0:
                maxsup = cpats[0]['targetsupport']
                maxratio = cpats[0]['supportratio']
            else:
                maxsup = 0.0
                maxratio = 0.0
            rdata['maxsup'].append(maxsup)
            rdata['maxratio'].append(maxratio)
        #break
    resdf = pd.DataFrame(rdata)



    medians = tmodel.get_layer('pat_binarize').layer_medians('activation_1')
    linfo = mlutil.conv_layer_info(basemodel, 'activation')
    #r = plot.build_feature_mapper(basemodel, ['activation'])
    bl = mlutil.make_output_nodes(basemodel, ['activation', 'activation_1'])

    plot.plot_receptive_field_set(basemodel, medians, ['activation_2-0', 'activation_2-1'], trainds.file_paths[:10], 'session/pattern')

    for i, (x, y) in enumerate(trainds):
        bx = bl(x)
        selfeats = tf.gather(bx[0], indices=[0, 1, 2], axis=-1)
        selfeats = tf.gather(selfeats, indices=[0, 1], axis=0)
        selfeats = tf.transpose(selfeats[0], perm=[2, 0, 1])
        fmap = selfeats[0]
        rmap = mlutil.map_receptive_field(fmap, bl.input_shape[1:], linfo)
        plot.plot_receptive_field(trainds.file_paths[0], rmap)
        gx = tf.gather(x, indices=[0, 1], axis=0)
        xp = r(gx)
        print(i)


    # Transactions
    outputlayers = ['activation_2']
    #tmodel = tm.build_transaction_model(basemodel, trainds, outputlayers)
    #tmodel.save('session/tmodel.keras')
    #tmodel = models.load_model('session/tmodel.keras')
    #trans = tmodel.predict(trainds)
    #bdf = tm.transactions_to_dataframe(tmodel, trans, trainds)
    #bdf.to_csv('session/bdf.csv')
    bdf = pd.read_csv('session/bdf.csv', index_col='index')

    #make_features()
    #eval()


    #run_model()

    #session = AzureSession()
    #session.open(working_dir='neural')
    #session.put('session/base.keras', 'neural/session')
    #session.put('session/tmodel.keras', 'neural/session')
    #session.put('session/bdf.csv', 'neural/session')
    #session.upload_function(run_model)
    #session.execute('python src/remote_spec.py')
    #session.get('neural/session/pmodel.zip', 'session')
    #session.close()

    #run_model()

    minsup = 0.5
    minsupratio = 1.1
    sel, notsel = data.filter_transactions(bdf, 'activation_2-', 1)
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
