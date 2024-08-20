import math

from tensorflow.keras import datasets, layers, models, backend, preprocessing, Input
import tensorflow as tf
import ssl
import numpy as np
import pandas as pd
import random
import cv2
from itertools import compress
import warnings
import spmf
import csv
from mlxtend.frequent_patterns import fpgrowth, fpmax
from mlutil import makeDebugModel, calcReceptiveField, calcReceptiveField2


def featureMapActivation(fmap):
    return np.max(fmap)


def featureActivations(outs, layernames):

    # Group fmap activations for all inputs
    colnames = []
    activations = []
    for oi in range(len(outs)):
        layer = outs[oi]
        numimages = outs[oi].shape[0]
        numfmaps = outs[oi].shape[-1]
        for fi in range(numfmaps):
            colnames.append(layernames[oi] + '-' + str(fi))
            fmaps = [featureMapActivation(layer[i, :, :, fi]) for i in range(numimages)]
            activations.append(fmaps)

    return colnames, activations


def layerNames(model):
    layernames = [l.name for l in model.layers if '2d' in l.name]
    return layernames


def entropy(ds):
    if len(ds) == 0:
        return 0.0
    correctpct = len([v for v in ds if v[1]]) / len(ds)
    incorrectpct = len([v for v in ds if not v[1]]) / len(ds)
    if correctpct == 0.0 or incorrectpct == 0.0:
        ent = 0.0
    else:
        ent = -1 * (correctpct * math.log2(correctpct) + incorrectpct * math.log2(incorrectpct))
    return ent


def cutPoints2(fmaps, iscorrect):
    cutpoints = []
    vals = list(zip(fmaps, iscorrect))
    tvals = np.unique(fmaps)
    testpoints = random.sample(range(len(tvals)), min(100, len(tvals)))
    mincut = -1
    mininfo = 99999
    d1size = 0
    d2size = 0
    for tp in testpoints:
        cutpoint = tvals[tp]
        d1 = [v for v in vals if v[0] <= cutpoint]
        d2 = [v for v in vals if v[0] > cutpoint]
        e1 = entropy(d1)
        e2 = entropy(d2)
        info = (len(d1) / len(vals)) * e1 + (len(d2) / len(vals)) * e2
        if info <= mininfo:
            mincut = cutpoint
            mininfo = info
            d1size = len(d1)
            d2size = len(d2)
    cutpoints.append(mincut)
    print('best info:' + str(mininfo) + ', cutpoint:' + str(mincut) + ', d1:' + str(d1size) + ', d2:' + str(d2size))
    return cutpoints


def cutPoints(fmaps, iscorrect):
    cutpoints = []
    vals = list(zip(fmaps, iscorrect))
    tvals = np.unique(fmaps)
    testpoints = random.sample(range(len(tvals)), min(100, len(tvals)))
    samplevals = [vals[i] for i in random.sample(range(len(vals)), min(1000, len(vals)))]
    mincut = -1
    mininfo = 99999
    d1size = 0
    d2size = 0
    for tp in testpoints:
        cutpoint = tvals[tp]
        d1 = [v for v in samplevals if v[0] <= cutpoint]
        d2 = [v for v in samplevals if v[0] > cutpoint]
        e1 = entropy(d1)
        e2 = entropy(d2)
        info = (len(d1) / len(samplevals)) * e1 + (len(d2) / len(samplevals)) * e2
        if info <= mininfo:
            mincut = cutpoint
            mininfo = info
            d1size = len(d1)
            d2size = len(d2)
    cutpoints.append(mincut)
    print('best info:' + str(mininfo) + ', cutpoint:' + str(mincut) + ', d1:' + str(d1size) + ', d2:' + str(d2size))
    return cutpoints


def discretize(fmaps, cutpoints):
    fbinned = []
    for f in fmaps:
        found = False
        for ci in range(len(cutpoints)):
            if f <= cutpoints[ci]:
                fbinned.append(ci)
                found = True
                break
        if not found:
            fbinned.append(len(cutpoints))
    return fbinned


def binarize(df):
    cuts = [[1] for i in range(len(df.columns) - 1)]

    # Get binary column names
    bincols = ['index']
    for ci in range(len(df.columns) - 1):
        for i in range(len(cuts[ci]) + 1):
            bincols.append(df.columns[ci] + '_' + str(i))
    bincols.append('iscorrect')

    # Binarize data
    brows = []
    for index, row in df.iterrows():
        rowbuf = [index]
        for ci in range(len(df.columns) - 1):
            v = row.iloc[ci]
            for i in range(len(cuts[ci]) + 1):
                rowbuf.append(v == i)
        rowbuf.append(row['iscorrect'] == 1)
        brows.append(rowbuf)

    bdf = pd.DataFrame(brows, columns=bincols)
    bdf.set_index('index', inplace=True)
    #bdf.index.name = 'index'
    return bdf


def evalModel(debugmodel):
    layernames = layerNames(debugmodel)
    outs = debugmodel.predict(test_images)
    ymaxpreds = [np.argmax(x) for x in outs[-1]]
    labels = [l[0] for l in test_labels]
    iscorrect = [ymaxpreds[i] == labels[i] for i in range(len(labels))]
    labelinfo = list(zip(ymaxpreds, labels, iscorrect))
    convouts = [o for o in outs if len(o.shape) == 4]  # get only CNN layers
    colnames, activations = featureActivations(convouts, layernames)

    cuts = [cutPoints(a, iscorrect) for a in activations]
    binned = [discretize(activations[i], cuts[i]) for i in range(len(activations))]
    d = {}
    for i in range(len(colnames)):
        d[colnames[i]] = binned[i]
    d['label'] = labels
    d['predicted'] = ymaxpreds
    d['iscorrect'] = [1 if c else 0 for c in iscorrect]
    df = pd.DataFrame(d)
    df.index.name = 'index'

    with open('cuts.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for i in range(len(colnames)):
            line = [colnames[i]] + cuts[i]
            writer.writerow(line)

    df.to_csv('test.csv')


def findMatches(ds, pattern):
    matches = []
    for index, row in ds.iterrows():
        print(index)
    return matches


def mineContrastPatterns(correct, incorrect):
    print('mine')

    correctsets = []
    for index, row in correct.iterrows():
        itemset = frozenset(compress(correct.columns, row))
        correctsets.append({'index': index, 'itemset': itemset})

    incorrectsets = []
    for index, row in incorrect.iterrows():
        itemset = frozenset(compress(incorrect.columns, row))
        incorrectsets.append({'index': index, 'itemset': itemset})

    dups = []
    for c in incorrect.columns:
        vals = incorrect[c].to_numpy()
        if (vals[0] == vals).all():
            dups.append(c)
    incorrect.drop(dups, axis=1, inplace=True)

    lookup = {}
    for i in range(len(incorrect.columns)):
        lookup[incorrect.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in incorrect.iterrows():
            itemset = [lookup[item] for item in list(compress(incorrect.columns, row))]
            writer.writerow(itemset)

    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[0.8, 5])
    fpclose.run()
    fpclose.parse_output()
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([incorrect.columns[int(itm)] for itm in items[:-2]])
        sup = int(items[-1]) / float(len(incorrect))
        patlist.append({'pattern': pat, 'incorrectsupport': sup, 'correctmatches': [], 'incorrectmatches': []})

    i = 0
    for p in patlist:
        print(str(i) + '/' + str(len(patlist)))
        i += 1
        count = 0
        for c in correctsets:
            if p['pattern'].issubset(c['itemset']):
                p['correctmatches'].append(c['index'])
                count += 1
        for c in incorrectsets:
            if p['pattern'].issubset(c['itemset']):
                p['incorrectmatches'].append(c['index'])

        p['correctsupport'] = count / float(len(correctsets))
        p['supportdiff'] = p['incorrectsupport'] - p['correctsupport']
        p['supportratio'] = p['incorrectsupport'] / p['correctsupport']

    patlist.sort(key=lambda x: x['incorrectsupport'] - x['correctsupport'])
    #for p in patlist:
    #    #if p['supportdiff'] > 0.05:
    #    print(p)
    return patlist


def plotReceptiveField(image, layers, fmap, color, cutlist):
    imga = image.copy()#cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fmin = np.min(fmap)
    fmax = np.max(fmap)
    #scaled = (((fmap - fmin) / (fmax - fmin)) * 255).astype(np.uint8)
    scaled = fmap.copy()
    for x in range(scaled.shape[0]):
        for y in range(scaled.shape[1]):
            if scaled[x, y] > fmax / 2.0: #cutlist[0]:
                scaled[x, y] = 255
            else:
                scaled[x, y] = 0
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    # Create receptive field mask
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            xfield, yfield = calcReceptiveField(x, y, layers)
            for xf in range(xfield[0], xfield[1] + 1):
                for yf in range(yfield[0], yfield[1] + 1):
                    if v > 0:
                        if color == 0:
                            imga[yf, xf] = (0, 255, 0)#(0, max(v, mask[yf, xf, 1]), 0)
                        elif color == 1:
                            imga[yf, xf] = (0, 0, 255) #(0, 0, max(v, mask[yf, xf, 1]))
                        #else:
                        #    mask[yf, xf] = (0, max(v, mask[yf, xf, 1]), 0)

    #combined = cv2.addWeighted(imga, .5, mask, 1.0, 0)
    combined = imga

    #cv2.imwrite('test_img.png', imga)
    #cv2.imwrite('test_mask.png', mask)
    #cv2.imwrite('test2.png', scaled)
    #cv2.imwrite('test_combined.png', combined)
    return combined, mask


def plotPatternReceptiveField(image, layeroutputs, layerinfo, pattern, cuts):
    #canvas = np.copy(image)
    canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = 0
    for item in pattern:
    #for j in range(2, 0, -1):
    #    item = list(pattern)[j-1]
        tokens = item.split('-')
        ftokens = tokens[1].split('_')
        layerindex = -1
        fmapindex = int(ftokens[0])
        cutlist = cuts[tokens[0] + '-' + str(fmapindex)]
        for li in range(len(layeroutputs)):
            if layeroutputs[li]['name'].startswith(tokens[0]):
                layerindex = li
                break
        o = layeroutputs[layerindex]['output'][:, :, fmapindex]
        layerset = layerinfo[:layerindex + 1]
        canvas, mask = plotReceptiveField(canvas, layerset, o, i, cutlist)
        cv2.imwrite('test_mask' + str(i) + '.png', mask)
        i += 1
        if i == 2:
            break
    #cv2.imwrite('test_combined3.png', canvas)
    return canvas


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    #a = model.get_layer(last_conv_layer_name)
    #grad_model = models.Model(
    #    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    #)
    img = preprocessing.image.img_to_array(img_array)
    img = np.expand_dims(img, axis=0)
    gradmodel = models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-1].output])
    #gradmodel.build((32, 32, 3))

    #inputs = Input((6, 6, 64))
    #convlayer = model.get_layer(last_conv_layer_name)
    #last_conv_outputs = convlayer(inputs)
    #x = last_conv_outputs
    #for i in range(6, len(model.layers)):
    #    x = model.layers[i](x)
    #output = x
    #gradmodel = models.Model(inputs, [last_conv_outputs, output])


    #inputs = Input((32, 32, 3))
    #convlayer = model.get_layer(last_conv_layer_name)
    #x = model.layers[1](model.inputs)
    #for i in range(2, len(model.layers)):
    #    x = model.layers[i](x)
    #output = x
    #gradmodel = models.Model(inputs, [convlayer.output, output])
    #print(1)
    #model(img)
    #convlayer = model.get_layer(last_conv_layer_name)
    #gradmodel = models.Model(model.inputs, [convlayer.output, model.layers[-1].output])
    #gradmodel(img)



    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:

        #    last_conv_layer_output, preds = grad_model(img_array)
        #    if pred_index is None:
        #        pred_index = tf.argmax(preds[0])
        #    class_channel = preds[:, pred_index]
        #tape.watch(img)
        #outs = model.predict(img_array)
        #last_conv_layer_output = outs[-4]
        #class_channel = outs[-1]
        img = tf.Variable(img)
        #tape.watch(img)
        last_conv_layer_output, preds = gradmodel(img)
        class_channel = preds[:, pred_index]
        #tape.watch(last_conv_layer_output)
        #tape.watch(class_channel)
        #tape.watch(gradmodel.trainable_variables)

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        #grads = tape.gradient(class_channel, last_conv_layer_output)
        grads = tape.gradient(class_channel, gradmodel.trainable_variables)
        print(1)
    #grads = tape.gradient(class_channel, img)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads[4], axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def newHeatmap(image, model, target):
    img_tensor = preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    heatmap_model = models.Model(inputs=model.inputs, outputs=[model.get_layer('conv2d_2').output, model.layers[-1].output])
    with tf.GradientTape as tape:
        conv_output, predictions = heatmap_model(img_tensor)
        t = np.argmax(predictions[0])
        loss = predictions[:, t]
        #grads = tape.gradient(loss, conv_output)
        grads = tape.gradient(loss, heatmap_model.trainable_variables)
        pooled_grads = backend.mean(grads[0], axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        print(1)


def stupidGradient(image, model):
    img_tensor = preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    heatmap_model = models.Model(inputs=model.inputs, outputs=[model.get_layer('conv2d_2').output, model.layers[-1].output])
    with tf.GradientTape() as tape:
        print(1)
        #tape.watch(img_tensor)
        #tape.watch(model.get_layer('conv2d_2').output)
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]
        #grads = tape.gradient(loss, heatmap_model.trainable_variables)
        grads = tape.gradient(loss, heatmap_model.trainable_variables[2])
        pooled_grads = backend.mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, heatmap_model.trainable_variables[2]), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        print(1)
        return heatmap

def visualize_class_activation_map(model, original_img):

    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    #img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.get_layer('conv2d_2')
    get_output =models.Model([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output(np.asarray([original_img]))
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
    target_class = 1
    for i, w in enumerate(class_weights[:, target_class]):
        cam += w * conv_outputs[i, :, :]
    print(1)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    #grad_model = models.Model(
    #    model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    #)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        #last_conv_layer_output, preds = grad_model(img_array)
        outs = model(img_array)
        preds = outs[-1]
        last_conv_layer_output = outs[4]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


ssl._create_default_https_context = ssl._create_unverified_context
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']






model = models.load_model('testmodel_softmax.keras', compile=True)
print(model.summary())

debugmodel = makeDebugModel(model)

#heatmap = make_gradcam_heatmap(test_images[0], model, 'conv2d_2', 3)
#newHeatmap(test_images[0], model, 3)
heatmap = stupidGradient(test_images[0], model)
#visualize_class_activation_map(model, test_images[0])
#make_gradcam_heatmap(np.asarray([test_images[0]]), debugmodel, 'conv2d_2', pred_index=3)

colormap=cv2.COLORMAP_VIRIDIS
alpha = 0.5
image = orig_test_images[0]
(w, h) = (image.shape[2], image.shape[1])
heatmap = cv2.resize(heatmap, (w, h))
numer = heatmap - np.min(heatmap)
denom = (heatmap.max() - heatmap.min()) + 1e-8
heatmap = numer / denom
heatmap = (heatmap * 255).astype("uint8")
heatmap = cv2.applyColorMap(heatmap, colormap)
output = cv2.addWeighted(orig_test_images[0], alpha, heatmap, 1 - alpha, 0)
print(1)
#m = 33
#outs = debugmodel.predict(np.asarray([test_images[m]]))
#layeroutputs = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]
#layerinfo = [{'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if '2d' in l.name]
#layerinfo.insert(0, {'kernel': debugmodel.layers[1].kernel_size, 'stride': debugmodel.layers[1].strides})

# first
#o = layeroutputs[0]['output'][:, :, 5]
#plotReceptiveField(orig_test_images[m], [layerinfo[0]], o)

# second
#o = layeroutputs[1]['output'][:, :, 5]
#plotReceptiveField(orig_test_images[m], [layerinfo[0], layerinfo[1]], o)

# third
#o = layeroutputs[2]['output'][:, :, 5]
#plotReceptiveField(orig_test_images[m], [layerinfo[0], layerinfo[1], layerinfo[2]], o)

# fourth
#o = layeroutputs[3]['output'][:, :, 5]
#plotReceptiveField(orig_test_images[m], [layerinfo[0], layerinfo[1], layerinfo[2], layerinfo[3]], o)

# fifth
#o = layeroutputs[4]['output'][:, :, 61]
#plotReceptiveField(orig_test_images[m], [layerinfo[0], layerinfo[1], layerinfo[2], layerinfo[3], layerinfo[4]], o)

#o = layeroutputs[4]['output'][:, :, 61]
#plotReceptiveField(orig_test_images[m], [layerinfo[0], layerinfo[1], layerinfo[2], layerinfo[3], layerinfo[4], layerinfo[5]], o)
#o = layeroutputs[-1]['output'][:, :, 61]
#plotReceptiveField(orig_test_images[m], layerinfo, o)


#evalModel(debugmodel)

cuts = {}
with open('cuts.csv') as infile:
    reader = csv.reader(infile, delimiter=' ')
    for row in reader:
        key = row[0]
        cutvals = [float(v) for v in row[1:]]
        cuts[key] = cutvals



df = pd.read_csv('test.csv', index_col=0)
di = class_names.index('dog')
ci = class_names.index('cat')
sel = df[df['label'].isin([di, ci])]
sel = df[df['predicted'].isin([di, ci])]
sel.drop(['label', 'predicted'], axis=1, inplace=True)
#cols = sel.columns[32:-1]   #32-63
cols = [sel.columns[i] for i in range(len(sel.columns)) if 'conv2d_1-' in sel.columns[i]]# or 'conv2d_2_' in sel.columns[i]]
#cols = sel.columns[:32]
#cols = cols[:30]
cols = [c for c in sel.columns if c not in cols and c != 'iscorrect']
sel.drop(cols, axis=1, inplace=True)
bds = binarize(sel)
correct = bds[bds['iscorrect']]
incorrect = bds[~bds['iscorrect']]
correct.drop('iscorrect', axis=1, inplace=True)
incorrect.drop('iscorrect', axis=1, inplace=True)
cpats = mineContrastPatterns(correct, incorrect)

pat = cpats[54]['pattern']
m = cpats[54]['incorrectmatches'][0]
outs = debugmodel.predict(np.asarray([test_images[m]]))
layeroutputs = [{'name': debugmodel.layers[i + 1].name, 'output': outs[i][0]} for i in range(len(outs)) if '2d' in debugmodel.layers[i + 1].name]
layerinfo = [{'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if '2d' in l.name]
#layerinfo.insert(0, {'kernel': debugmodel.layers[1].kernel_size, 'stride': debugmodel.layers[1].strides})

for i in range(len(cpats)):
    pat = cpats[i]['pattern']
    m = cpats[i]['incorrectmatches'][0]
    img = plotPatternReceptiveField(orig_test_images[m], layeroutputs, layerinfo, pat, cuts)
    cv2.imwrite('temp/' + str(i) + '.png', img)
for item in pat:
    tokens = item.split('-')
    ftokens = tokens[1].split('_')
    layerindex = -1
    fmapindex = int(ftokens[0])
    for li in range(len(layeroutputs)):
        if layeroutputs[li]['name'].startswith(tokens[0]):
            layerindex = li
            break
    o = layeroutputs[layerindex]['output'][:, :, fmapindex]
    layerset = layerinfo[:layerindex + 1]
    plotReceptiveField(orig_test_images[m], layerset, o)
    print(1)


#o = l[0]['output'][:, :, 1]
o = layeroutputs[-1]['output'][:, :, 61]
#plotReceptiveField(orig_test_images[imageindex], [ll[0]], o)
plotReceptiveField(orig_test_images[m], layerinfo, o)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

model.save('testmodel_softmax.keras')
