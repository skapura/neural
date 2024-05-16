from keras.utils import image_dataset_from_directory
from keras.src.models import Functional
from keras import layers, models
import tensorflow as tf
import numpy as np
from tensorflow import data as tf_data
import keras
import pandas as pd
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


trainds = image_dataset_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256))
valds = image_dataset_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256))
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

#bdf.loc[:, 'iscorrect'] = pd.Series(iscorrect, index=bdf.index)
bdf['iscorrect'] = iscorrect

correct = bdf[bdf['iscorrect']]
incorrect = bdf[~bdf['iscorrect']]
correct.drop('iscorrect', axis=1, inplace=True)
incorrect.drop('iscorrect', axis=1, inplace=True)
pats.mineContrastPatterns(correct, incorrect)

ranges = mlutil.getLayerOutputRange(model, ['activation', 'activation_1', 'prediction'], valds)

for e in valds:
    print(e[0][0])
    print(1)


model = models.load_model('largeimage.keras', compile=True)


print(1)
