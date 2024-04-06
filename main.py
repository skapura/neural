import keras.layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import math
import ssl
import cv2
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
# numpy: height X width
# cv2: width X height

def makeDebugModel(model, onlyconv=False):
    if onlyconv:
        outputs = [layer.output for layer in model.layers if type(layer) is keras.layers.Conv2D]
    else:
        outputs = [layer.output for layer in model.layers]
    debugmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return debugmodel


def evalModel(X, y, model):
    ypreds = model.predict(X)
    #correct = list()
    #incorrect = list()
    #for i in range(len(y)):
    #    yp = ypreds[i]
    #    maxclass = np.argmax(yp)
    #    correctclass = y[i][0]
    #    if y[i][0] == maxclass:
    #        correct.append(i)
    #    else:
    #        incorrect.append(i)
    #    #print(1)


    ymaxpreds = [np.argmax(x) for x in ypreds]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], y), ymaxpreds)]
    correct = [i for i, elem in enumerate(eval) if elem]
    incorrect = [i for i, elem in enumerate(eval) if not elem]
    return correct, incorrect


def selectImages(images, labels, indexlist, condition=None):
    selected = [(images[x], labels[x][0]) for x in indexlist]
    if condition is not None:
        selected = [x for x in selected if condition(x)]
    return selected


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

    print(1)


def renderFeatureMaps(maps):
    fh = 30#maps.shape[0]
    fw = 30#maps.shape[1]
    xmaps = 30
    numfilters = maps.shape[-1]
    ymaps = math.ceil(numfilters / xmaps)
    f_min, f_max = maps.min(), maps.max()
    scaled = np.interp(maps, [f_min, f_max], [0, 255]).astype(np.uint8)
    (_, lh), _ = cv2.getTextSize('0', cv2.FONT_HERSHEY_PLAIN, 1, 1)
    img = np.full(((fh + lh + 2) * ymaps + 1, (fw + 1) * xmaps + 1), 255, np.uint8)
    x = 1
    y = 0
    for i in range(0, numfilters):
        filter = scaled[:, :, i]
        #r = np.rollaxis(filter, 1)  # swap HxW->WxH for cv2
        (_, lh), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_PLAIN, 1, 1)
        img = cv2.putText(img, str(i), (x, y + lh + 1), cv2.FONT_HERSHEY_PLAIN, 1, 0, 1)
        r = cv2.resize(filter, (fh, fw), interpolation=cv2.INTER_NEAREST_EXACT)
        img[y + lh + 2:y + r.shape[0] + lh + 2, x:x + r.shape[1]] = r
        if (i + 1) % xmaps == 0:
            x = 1
            y += fh + lh + 2
        else:
            x += fw + 1

    return img


def rotate3(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_mat


def renderClassScores(classes, output):
    scaler = MinMaxScaler(feature_range=(-100, 100))
    rout = scaler.fit_transform(output.reshape(-1, 1)).flatten()
    maxindex = max(enumerate(output), key=lambda x: x[1])[0]

    # Get max text rendered length
    th = 0
    for c in classes:
        (lw, _), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        th = max(th, lw)

    canvas = np.full((th + 200, len(classes) * 20 - 5, 3), 255, np.uint8)

    # Render bar plot
    x = 0
    i = 0
    for o in rout:
        y = 100 - int(o)
        y1 = min(y, 100) - 1
        y2 = max(y, 100) - 1
        if i == maxindex:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(canvas, (x, y1), (x + 15, y2), color, -1)
        x += 20
        i += 1
    cv2.line(canvas, (0, 99), (canvas.shape[0] - 1, 99), (0, 0, 0), 2)

    # Render class labels
    y = 200
    x = 0
    for c in classes:
        (lw, lh), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        tc = np.full((lh, lw, 3), 255, np.uint8)
        tc = cv2.putText(tc, c, (0, lh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        tc = rotate3(tc, 90)
        canvas[y:y + tc.shape[0], x:x + tc.shape[1]] = tc
        x += 20

    return canvas


def renderSummary(image, title, label, maps, classes, output):
    rmaps = [renderFeatureMaps(m['output']) for m in maps]
    rclasses = renderClassScores(classes, output)

    (_, lh), _ = cv2.getTextSize('0', cv2.FONT_HERSHEY_PLAIN, 1, 1)
    h = sum(m.shape[0] for m in rmaps) + rclasses.shape[0] + image.shape[0] + len(rmaps) * (lh + 2) + 6
    w = max(m.shape[1] for m in rmaps)
    canvas = np.full((h, w, 3), 255, np.uint8)

    #renderedmaps = renderFeatureMaps(maps)
    #h = renderedmaps.shape[0] + image.shape[0] + 6
    #w = renderedmaps.shape[1]
    #canvas = np.full((h, w, 3), 255, np.uint8)

    x = 0
    y = 1
    canvas[y:y + image.shape[0], x + 1:x + image.shape[1] + 1] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, title + ' - ' + label, (x + image.shape[1] + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    #(lw, lh), _ = cv2.getTextSize('0123456789', cv2.FONT_HERSHEY_PLAIN, 1, 1)
    #tc = np.full((lh, lw, 3), 255, np.uint8)
    #tc = cv2.putText(tc, '0123456789', (0, lh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    #M = cv2.getRotationMatrix2D((lw / 2, lh / 2), 90, 1)
    #tc = cv2.warpAffine(tc, M, (lw, lh))
    #canvas[100:100 + tc.shape[0], 100:100 + tc.shape[1]] = tc

    #rows, cols, _ = tc.shape
    #M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #tc = cv2.warpAffine(tc, M, (rows, cols))
    #tc = rotate3(tc, 90)
    #cv2.imwrite('text.png', tc)
    #canvas[100:100 + tc.shape[0], 100:100 + tc.shape[1]] = tc

    y += image.shape[0] + 5
    for i in range(len(rmaps)):
        rgbmap = cv2.cvtColor(rmaps[i], cv2.COLOR_GRAY2RGB)
        #mapy = y + image.shape[0] + 5
        title = maps[i]['name'] + ' - ' + str(maps[i]['output'].shape[0]) + 'x' + str(maps[i]['output'].shape[1]) + 'x' + str(maps[i]['output'].shape[2])
        canvas = cv2.putText(canvas, title, (1, y + lh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        y += lh + 1
        canvas[y:y + rgbmap.shape[0], x:x + rgbmap.shape[1]] = rgbmap
        y += rgbmap.shape[0] + 1

    canvas[y:y + rclasses.shape[0], 1:1 + rclasses.shape[1]] = rclasses
    cv2.imwrite('canvas.png', canvas)
    print(1)


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

ti = 0
outs = debugmodel.predict(np.asarray([test_images[ti]]))

l = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]



layer1 = outs[3][0]
#renderSummary(orig_test_images[0], class_names[test_labels[0][0]], layer1)
#renderSummary(orig_test_images[ti], class_names[test_labels[ti][0]], l, class_names, outs[-1][0])

correct, incorrect = evalModel(test_images, test_labels, model)

wrong_test_images = test_images[incorrect]
wrong_test_labels = test_labels[incorrect]

#wrongimage = orig_test_images[incorrect[1]]
#correctlabel = class_names[test_labels[incorrect[1]][0]]
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
