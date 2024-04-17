import keras.layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import math
import ssl
import cv2
import shutil
import os
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from mlutil import makeDebugModel, evalModel
from plot import plotReceptiveField, renderSummary
# numpy: height X width
# cv2: width X height


def selectImages(images, labels, indexlist, condition=None):
    selected = [(images[x], labels[x][0]) for x in indexlist]
    if condition is not None:
        selected = [x for x in selected if condition(x)]
    return selected


def collectImageSet(images, labels, classnames):
    shutil.rmtree('imageset')
    os.mkdir('imageset')
    for img in images:
        path = str(img).zfill(5) + ' - ' + classnames[labels[img][0]] + '.png'
        shutil.copy('images/' + path, 'imageset')


def renderFilters(maps):
    fw = 18
    fh = 18
    if fw is None and fh is None:
        fh = maps.shape[0]
        fw = maps.shape[1]
    xmaps = 15
    numfilters = maps.shape[-1]
    ymaps = math.ceil(numfilters / xmaps)
    f_min, f_max = maps.min(), maps.max()
    scaled = np.interp(maps, [f_min, f_max], [0, 255]).astype(np.uint8)
    #img = np.zeros((maps.shape[1], maps.shape[0], 3), np.uint8)
    img = np.zeros(((fh + 1) * ymaps - 1, (fw + 1) * xmaps - 1, 3), np.uint8)
    x = 0
    y = 0
    for i in range(0, numfilters):
        filter = scaled[:, :, :, i]
        #r = np.rollaxis(filter, 1)  # swap HxW->WxH for cv2
        r = cv2.resize(filter, (fh, fw), interpolation=cv2.INTER_NEAREST_EXACT)
        img[y:y + r.shape[0], x:x + r.shape[1]] = r
        if (i + 1) % xmaps == 0:
            x = 0
            y += fw + 1
        else:
            x += fw + 1

    #filter = scaled[:, :, :, 0]
    #r = np.rollaxis(filter, 1)  # swap HxW->WxH for cv2
    #r = cv2.resize(r, (fw, fh), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imwrite('filters.png', img)



ssl._create_default_https_context = ssl._create_unverified_context

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#for i in range(len(orig_test_images)):
#    img = cv2.cvtColor(orig_test_images[i], cv2.COLOR_RGB2BGR)
#    imgclass = class_names[test_labels[i][0]]
#    cv2.imwrite('images/' + str(i).zfill(5) + ' - ' + imgclass + '.png', img)



model = models.load_model('testmodel.keras')
print(model.summary())

debugmodel = makeDebugModel(model)
print(backend.image_data_format())
filters, bias = debugmodel.layers[1].get_weights()
#renderFilters(filters)

#ti = 0
#outs = debugmodel.predict(np.asarray([test_images[ti]]))

#l = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]


correct, incorrect, wronganswers = evalModel(test_images, test_labels, model)

di = class_names.index('dog')
ci = class_names.index('cat')
correctdoglist = list()
correctcatlist = list()
incorrectdoglist = list()
incorrectcatlist = list()
totals = dict()
for a in wronganswers:
    key = str(a['correct']) + '_' + str(a['predicted'])
    if key in totals:
        totals[key] = totals[key] + 1
    else:
        totals[key] = 1

    if a['correct'] == di and a['predicted'] == ci:
        incorrectdoglist.append(a['index'])
    elif a['correct'] == ci and a['predicted'] == di:
        incorrectcatlist.append(a['index'])

for t in totals:
    classes = t.split('_')
    print(class_names[int(classes[0])] + '->' + class_names[int(classes[1])] + ': ' + str(totals[t]))

correctdog = 0
correctcat = 0
for c in correct:
    if test_labels[c][0] == di:
        correctdog += 1
        correctdoglist.append(c)
    elif test_labels[c][0] == ci:
        correctcat += 1
        correctcatlist.append(c)
print('correct dog: ' + str(correctdog))
print('correct cat: ' + str(correctcat))

wrong_test_images = test_images[incorrect]
wrong_test_labels = test_labels[incorrect]

idx = class_names.index('dog')
matches = [x for x in wronganswers if x['correct'] == idx]

imageindex = matches[0]['index']
outs = debugmodel.predict(np.asarray([test_images[imageindex]]))
layeroutputs = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]

layerinfo = [{'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if '2d' in l.name]
layerinfo.insert(0, {'kernel': debugmodel.layers[1].kernel_size, 'stride': debugmodel.layers[1].strides})

#o = l[0]['output'][:, :, 1]
o = layeroutputs[-1]['output'][:, :, 0]
#fieldx, fieldy = mlutil.calcReceptiveField(3, 3, ll)
#plotReceptiveField(orig_test_images[imageindex], [ll[0]], o)
plotReceptiveField(orig_test_images[imageindex], layerinfo, o)

renderSummary(orig_test_images[imageindex], str(imageindex), class_names[test_labels[imageindex][0]], layeroutputs, class_names, outs[-1][0])

collectImageSet(incorrect, test_labels, class_names)
incorrectindex = incorrect[2]
title = str(incorrectindex)
wrongimage = orig_test_images[incorrectindex]
correctlabel = class_names[test_labels[incorrectindex][0]]
outs = debugmodel.predict(np.asarray([test_images[incorrectindex]]))
#outs2 = debugmodel.predict(np.asarray([wrongimage]))
l = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]
renderSummary(wrongimage, title, correctlabel, l, class_names, outs[-1][0])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
f = filters[:,:,:,0]
fig = plt.figure(figsize=(8,8))
plt.imshow(f[:,:,0],cmap='gray')
plt.xticks([])
plt.yticks([])
#plot the filters
plt.show()


#outputs = debugmodel.predict(test_images)
# pr = model.predict(test_images)
correct, incorrect = evalModel(test_images, test_labels, model)
#layerinfo = debugmodel.predict(test_images[incorrect])
sel = selectImages(test_images, test_labels, incorrect)
layerinfo = debugmodel.predict(np.asarray([x[0] for x in sel]))

predoutputs = [x[0] for x in layerinfo]

outs = list()
for layer in model.layers:
    #if type(layer) is keras.layers.Conv2D:
    print(layer.name)
    #f, b = layer.get_weights()
    outs.append(layer.output)

k2 = models.Model(inputs=model.inputs, outputs=outs)
ylayers = k2.predict(test_images[:3])

clayer =4
imageindex = 2
maps = ylayers[clayer][imageindex]

m = 32 / 16
m2 = 64 / 16

numrows = int(maps.shape[2] / 16)

#plt.figure(figsize=(10,5))
fig, ax = plt.subplots(numrows + 1, 16, figsize=(15, 5))
fig.suptitle(class_names[test_labels[imageindex][0]])
r = 0
c = 0
for i in range(maps.shape[2]):
    fmap = maps[:, :, i] * 255
    #plt.subplot(3, 16, i + 1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #ax[r, c].xticks([])
    #ax[r, c].yticks([])
    #ax[r, c].grid(False)
    ax[r, c].imshow(fmap, cmap='gray', interpolation='nearest', aspect='equal')
    #ax[r, c].imshow(fmap, cmap='gray')
    ax[r, c].set_xticks([])
    ax[r, c].set_yticks([])
    c += 1
    if c == 16:
        r += 1
        c = 0
#plt.subplot(3, 16, maps.shape[2])
#plt.imshow(test_images[0])
#ax[2, 0].imshow(test_images[0], interpolation='nearest', aspect='auto')
ax[numrows, 0].imshow(test_images[imageindex])
ax[numrows, 0].set_xticks([])
ax[numrows, 0].set_yticks([])
for i in range(1, 16):
    #ax[2, i].imshow(np.zeros(shape=(30, 30)), cmap='gray', interpolation='nearest', aspect='auto')
    ax[numrows, i].imshow(np.zeros(shape=(maps.shape[0], maps.shape[1])), cmap='gray')
    ax[numrows, i].set_xticks([])
    ax[numrows, i].set_yticks([])
#plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

plt.show()

fmap = ylayers[0][0][:, :, 0] * 255
print(fmap)
plt.imshow(fmap, cmap='gray')

x = train_images[0]
y = model.predict(test_images[:3])


wrong = list()
for i in range(len(y)):
    index = max(enumerate(y[i]), key=lambda x: x[1])[0]
    if index != test_labels[i][0]:
        wrong.append(i)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

model.save('testmodel.keras')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

plt.show()
