from tensorflow.keras import datasets, layers, models, callbacks
from keras.src.models import Functional
import tensorflow as tf
import numpy as np
import cv2
import ssl
import warnings
import pandas as pd
from mlutil import makeDebugModel, makeLayerOutputModel, heatmap, overlayHeatmap, calcReceptiveField, layerIndex
import mlutil
from cifar10vgg import build_vgg16


def buildModel(train_images, train_labels, test_images, test_labels):
    img_input = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3))(img_input)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, name='prediction', activation='softmax')(x)
    model = Functional(img_input, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save('testmodel_gap_func.keras')

    #model = models.Sequential()
    #model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.GlobalAveragePooling2D())
    ## model.add(layers.Flatten())
    ## model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(10, activation='softmax'))
    #model.summary()
    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    #history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    #model.save('testmodel_gap.keras')



def evalModel(model, images, labels):
    labels = np.squeeze(labels)
    outs = model.predict(images)
    o = outs[-1] if type(outs) is list else outs
    ymaxpreds = [np.argmax(x) for x in o]
    iscorrect = [ymaxpreds[i] == labels[i] for i in range(len(labels))]
    results = pd.DataFrame({'predicted': ymaxpreds, 'label': labels, 'iscorrect': iscorrect})
    results.index.name = 'index'
    return results


ssl._create_default_https_context = ssl._create_unverified_context
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#buildModel(train_images, train_labels, test_images, test_labels)
model = models.load_model('testmodel_gap_func.keras', compile=True)
debugmodel = makeDebugModel(model)
#model = build_vgg16()

#results = evalModel(model, test_images, test_labels)
#di = class_names.index('dog')
#ci = class_names.index('cat')
#results = results[results['label'].isin([di, ci])]
#results = results[results['predicted'].isin([di, ci])]
#correct = results[results['iscorrect']]
#incorrect = results[~results['iscorrect']]

#index = incorrect.index.values[0]
#index = results.index.values[0]





#norms = np.zeros((3, 3))
#a = outs[1][:, :, :, 0]
#mn = np.min(a)
#mx = np.max(a)
#normed = (a - mn) / (mx - mn)

outputlayers = ['activation', 'prediction']
lastlayer = 'activation_2'

trans = mlutil.featuresToDataFrame(model, outputlayers, lastlayer, test_images)


franges = mlutil.getLayerOutputRange(model, outputlayers, test_images)



image = test_images[0]
heats = mlutil.combineHeatAndFeatures(model, franges, outputlayers, 'activation_2', image)

trans = []
head = []
for layerindex in range(len(heats)):
    for fi in range(len(heats[layerindex])):
        head.append(outputlayers[layerindex] + '-' + str(fi))
        v = heats[layerindex][fi].max()
        trans.append(v)



sel = []
for h in heats:
    layer = []
    for f in h:
        if f.max() > 2.0:
            layer.append(h)
    sel.append(layer)
print(1)
convinfo = [{'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if 'conv2d' in l.name or 'max_pooling2d' in l.name]
start, end = calcReceptiveField(0, 0, [convinfo[0]])
#outs = debugmodel.predict(np.asarray([test_images[index]]))

for index, row in correct.iterrows():
    heatmap, himg = heatmap(np.asarray([test_images[index]]), model, 'activation_2', row['predicted'])
    heatimage = overlayHeatmap(np.asarray([orig_test_images[index]]), heatmap)
    hu = np.unique(heatmap)
    hi = np.unique(heatimage)
    print(1)
    #cv2.imwrite('heat/correct/' + str(index).zfill(5) + '.png', heatimage)
for index, row in incorrect.iterrows():
    heatmap = heatmap(np.asarray([test_images[index]]), model, 'activation_2', row['predicted'])
    heatimage = overlayHeatmap(np.asarray([orig_test_images[index]]), heatmap)
    cv2.imwrite('heat/incorrect/' + str(index).zfill(5) + '.png', heatimage)
print(1)
