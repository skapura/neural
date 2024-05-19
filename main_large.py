from keras.utils import image_dataset_from_directory
from keras.src.models import Functional
from keras import layers, models
import tensorflow as tf
import numpy as np
from tensorflow import data as tf_data
import keras
import pickle
import pandas as pd
import cv2
import mlutil
import patterns as pats


def buildModel():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


def generateTrans(model, images, labels, class_names, outputlayers):
    labels = labels.squeeze()
    selectedindexes = filterLabels(labels, class_names, ['church', 'golf_ball'])
    selectedimages = [images[i] for i in selectedindexes]
    selectedlabels = [labels[i] for i in selectedindexes]

    outputlayers = ['activation', 'prediction']
    lastlayer = 'activation_2'

    franges = mlutil.getLayerOutputRange(model, outputlayers, images)
    trans = mlutil.featuresToDataFrame(model, outputlayers, lastlayer, franges, selectedindexes, selectedimages,
                                       selectedlabels)
    t = trans[trans['predicted'].isin([di, ci])]
    t.to_csv('trans.csv')


def evalMatches(model, trans):
    with open('patterns.pkl', 'rb') as f:
        cpats = pickle.load(f)
    matches = set()
    cmatch = set()
    for p in cpats[:10]:
        matches.update(p['incorrectmatches'])
        cmatch.update(p['correctmatches'])
        #print(p['supportdiff'])

    #for p in cmatch:
    #    path = trans.loc[0]['imagepath']
    #    img = cv2.imread(path)
    #    img = cv2.resize(img, (256, 256))
    #    label = trans.loc[0]['label']
    #    pred = trans.loc[0]['predicted']
    #    outs = model.predict(np.asarray([img]))
    #    print(1)

    batchindex = 0
    #for imagebatch, labelbatch in valds:
    for index in matches:
        img = cv2.imread(trans.loc[index]['imagepath'])
        img = cv2.resize(img, (256, 256))
        #outs = model.predict(np.asarray([img]))
        h, _ = mlutil.heatmap(np.asarray([img]), model, 'activation_3', trans.loc[index]['predicted'])
        heatout = mlutil.overlayHeatmap(np.asarray([img]), h)
        cv2.imwrite('sessionimg/heat_' + str(index).zfill(5) + '_' + str(trans.loc[index]['label']) + '_' + str(trans.loc[index]['predicted']) + '.png', heatout)
        cv2.imwrite('sessionimg/heat_' + str(index).zfill(5) + '_' + str(trans.loc[index]['label']) + '_' + str(trans.loc[index]['predicted']) + '_orig.png', img)

    print(1)


trainds = image_dataset_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=False)
valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
#trainds = trainds.prefetch(tf_data.AUTOTUNE)
#valds = valds.prefetch(tf_data.AUTOTUNE)
model = models.load_model('largeimage.keras', compile=True)
#model = buildModel()
#model.fit(trainds, epochs=10, validation_data=valds)
#model.save('largeimage.keras')

#outs = model.predict(valds)
#valds = valds.take(1)
franges = None
batchnum = 1
outputlayers = ['activation', 'activation_1', 'prediction']
lastlayer = ['activation_3']
#for image, label in valds:
#    print(batchnum)
#    batchnum += 1
#    franges = mlutil.getLayerOutputRange(model, outputlayers, image, franges)

#trans = mlutil.featuresToDataFrame(model, outputlayers, valds)
#trans = (trans - trans.min()) / (trans.max() - trans.min())
#trans.to_csv('test.csv')

trans = pd.read_csv('test.csv', index_col='index')
evalMatches(model, trans)

# Drop columns that are all zero
droplist = []
for c in trans.columns[:-2]:
    mx = trans[c].max()
    if mx <= 0.05:
        droplist.append(c)
print(len(trans.columns))
trans.drop(droplist, axis=1, inplace=True)
print(len(trans.columns))

iscorrect = (trans['predicted'] == trans['label']).to_numpy()
cutpoints = []
infolist = []
trans.drop(['predicted', 'label', 'imagepath'], axis=1, inplace=True)

for c in trans.columns:
    print(c)
    vals = trans[c].to_numpy()
    cuts, info = pats.cutPoints(vals, iscorrect)
    cutpoints.append(cuts)
    infolist.append(info)
bds = pats.discretize(trans, cutpoints)
bdf = pats.binarize(bds, cutpoints)


bdf['iscorrect'] = iscorrect
correct = bdf[bdf['iscorrect']]
incorrect = bdf[~bdf['iscorrect']]
correct = correct[correct.columns[:-1]]
incorrect = incorrect[incorrect.columns[:-1]]
#cpats = pats.mineContrastPatterns(correct, incorrect, 0.8, 1.3)
#with open('patterns.pkl', 'wb') as f:
#    pickle.dump(cpats, f)



ranges = mlutil.getLayerOutputRange(model, ['activation', 'activation_1', 'prediction'], valds)

for e in valds:
    print(e[0][0])
    print(1)


model = models.load_model('largeimage.keras', compile=True)


print(1)
