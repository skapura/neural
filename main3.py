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

franges = mlutil.getLayerOutputRange(model, ['conv2d', 'activation', 'prediction'], test_images)

outs = debugmodel.predict(np.asarray([test_images[0]]))
predicted = np.argmax(outs[-1])
heatmap, himg = heatmap(np.asarray([test_images[0]]), model, 'activation_2', predicted)
#heatimage = overlayHeatmap(np.asarray([orig_test_images[0]]), heatmap)
convinfo = [{'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if 'conv2d' in l.name or 'max_pooling2d' in l.name]

h = np.unique(himg)
med = np.median(himg)
himg[himg < .7] = 0
idx = layerIndex(model, 'activation')
layerindex = 2
fmaps = outs[layerindex][0]
heats = []
for fi in range(fmaps.shape[2]):
    if franges[fi][1] == 0.0:
        currfmap = fmaps[:, : fi]
    else:
        currfmap = (fmaps[:, :, fi] - franges[fi][0]) / (franges[fi][1] - franges[fi][0])
    heatfeatmap = np.zeros(currfmap.shape)
    for x in range(currfmap.shape[1]):
        for y in range(currfmap.shape[0]):
            xrange, yrange = calcReceptiveField(x, y, [convinfo[0]])
            v = currfmap[y, x]

            # Use all vals in receptive field on heatmap
            hr = himg[yrange[0]:yrange[1]+1, xrange[0]:xrange[1]+1]
            heatfeatmap[y, x] = currfmap[y, x] * np.sum(hr)

            # Only use center val in receptive field on heatmap (assume 3x3 kernel)
            #heatfeatmap[y, x] = currfmap[y, x] * himg[yrange[0] + 1, xrange[0] + 1]
    heats.append(heatfeatmap)
sel = []
for h in heats:
    if h.max() >= 2.0:
        sel.append(h)
print(1)
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
