import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from keras import layers, models
import numpy as np
import mlutil
import keras
import cv2
import pandas as pd
import math
import patterns as pats
import tensorflow as tf
from keras.src.models import Functional


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
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


def scale(x, in_min, in_max, out_min, out_max):
    n = (x - in_min) * (out_max - out_min)
    d = (in_max - in_min) + out_min
    if math.isnan(n) or math.isnan(d) or d == 0.0:
        return 0.0
    else:
        #print(n)
        #print(d)
        return n / d


def scaleframe(df, out_min, out_max):
    mx = df.iloc[:, :-3].max()
    s = df.iloc[:, :-3].to_numpy()

    for c in range(s.shape[1]):
        maxval = mx.iloc[c]
        for r in range(s.shape[0]):
            v = s[r, c]
            s[r, c] = scale(v, 0.0, maxval, out_min, out_max)
    newdf = pd.DataFrame(s, columns=df.columns[:-3])
    newdf.index = df.index
    newdf = pd.concat([newdf, df['label']], axis=1)
    return newdf


def renderModel(model, activations, color):
    nodesize = (20, 20)
    layerdepth = 20
    nodesep = 5
    alayers = [l for l in model.layers if 'activation' in l.name]
    maxnodes = max([l.input.shape[-1] for l in alayers])
    canvas = np.full(((layerdepth + nodesize[0]) * len(alayers), maxnodes * (nodesep + nodesize[1]), 3), (255, 255, 255), dtype=np.uint8)

    # Render edges
    for y in range(len(alayers) - 1):
        yp = y * (layerdepth + nodesize[0])
        layerwidth = alayers[y].input.shape[-1] * (nodesep + nodesize[1])
        xoffset = int((canvas.shape[1] - layerwidth) / 2)
        elayerwidth = alayers[y + 1].input.shape[-1] * (nodesep + nodesize[1])
        xeoffset = int((canvas.shape[1] - elayerwidth) / 2)
        for x in range(alayers[y].input.shape[-1]):
            xp = xoffset + x * (nodesep + nodesize[1])
            ype = (y + 1) * (layerdepth + nodesize[0])
            start = (xp + int(nodesize[0] / 2), yp + nodesize[1])
            for xn in range(alayers[y + 1].input.shape[-1]):
                xpe = xeoffset + xn * (nodesep + nodesize[1])
                end = (xpe + int(nodesize[0] / 2), ype)
                #cv2.line(canvas, start, end, (0, 0, 0), 1)

    # Render nodes
    for y in range(len(alayers)):
        yp = y * (layerdepth + nodesize[0])
        layerwidth = alayers[y].input.shape[-1] * (nodesep + nodesize[1])
        xoffset = int((canvas.shape[1] - layerwidth) / 2)
        for x in range(alayers[y].input.shape[-1]):
            name = alayers[y].name + '-' + str(x)
            a = int(activations[name])
            xp = xoffset + x * (nodesep + nodesize[1])
            if color == 0:
                c = (a, 0, 0)
            elif color == 1:
                c = (0, a, 0)
            elif color == 2:
                c = (0, 0, a)
            cv2.rectangle(canvas, (xp, yp), (xp + nodesize[1], yp + nodesize[0]), c, -1)

    return canvas
    #cv2.imwrite('test.png', canvas)
    #print(10)


def renderlabel(model, trans, label):
    s = trans.loc[trans['label'] == label].iloc[:, :-1]
    mx = trans.iloc[:, :-1].max()
    mm = s.mean()
    #for i in range(len(mm)):
    #    mm.iloc[i] = scale(mm.iloc[i], 0.0, mx.iloc[i], 0, 255)
    #    if math.isnan(mm.iloc[i]):
    #        mm.iloc[i] = 0.0
    canvas = renderModel(model, mm, label)
    return canvas


def renderlabel2(model, trans, label):
    s = trans.loc[trans['label'] == label].iloc[:, :-3]
    mx = trans.iloc[:, :-3].max()
    mm = s.mean()
    for i in range(len(mm)):
        mm.iloc[i] = scale(mm.iloc[i], 0.0, mx.iloc[i], 0, 255)
        if math.isnan(mm.iloc[i]):
            mm.iloc[i] = 0.0
    canvas = renderModel(model, mm, label)
    return canvas


def renderavg(trans):
    tscaled = scaleframe(trans, 0, 255)
    c0 = renderlabel(model, tscaled, 0)
    c1 = renderlabel(model, tscaled, 1)
    c2 = renderlabel(model, tscaled, 2)
    canvas = np.full((c0.shape[0] * 3, c0.shape[1], 3), (255, 255, 255), dtype=np.uint8)
    canvas[:c0.shape[0], :c0.shape[1], :] = c0
    canvas[c0.shape[0]:c0.shape[0] * 2, :c0.shape[1], :] = c1
    canvas[c0.shape[0] * 2:c0.shape[0] * 3, :c0.shape[1], :] = c2
    cv2.imwrite('test.png', canvas)


def renderactivations(outs, outpath):
    vals = outs[:-1]
    label = int(outs[-1])
    canvas = renderModel(model, vals, label)
    cv2.imwrite(outpath, canvas)


def makebinary(trans, threshold):
    bvals = {}
    for col, vals in trans.items():
        if col == 'label':
            continue
        vlist = []
        for v in vals:
            vlist.append(1 if v >= threshold else 0)
        bvals[col] = vlist
    bvals['label'] = trans['label']
    bdf = pd.DataFrame(bvals)
    bdf.index = trans.index
    return bdf


def displaycounts(binned):
    for c in binned.columns[:-1]:
        v = binned.groupby('label')[c].value_counts()
        print(c)
        for i in range(len(v)):
            idx = v.index[i]
            if idx[1] == 1:
                a = v.iloc[i]
                print(str(idx[0]) + ', ' + str(a))


def nodeig(trans, node):
    labels = trans['label'].unique()
    labelcounts = trans['label'].value_counts()
    infod = 0.0
    for lc in labelcounts:
        p = lc / len(trans)
        infod -= p * math.log2(p)

    nodecounts = trans[[node, 'label']].groupby('label', as_index=False).value_counts()
    infonode = 0.0
    for l in labels:
        labelnodecounts = nodecounts[nodecounts['label'] == l]
        vals = labelnodecounts[node].unique()
        infoa = 0.0
        classtotal = labelcounts[l]
        for v in vals:
            lnc = labelnodecounts[labelnodecounts[node] == v]['count'].iloc[0]
            p = lnc / float(classtotal)
            infoa -= p * math.log2(p)

        pc = classtotal / float(len(trans))
        infonode += pc * infoa

    return infod - infonode


def infogain(trans):
    vals = list()
    for c in trans.columns[:-1]:
        info = nodeig(trans, c)
        vals.append(info)
    infos = pd.Series(vals, index=trans.columns[:-1])
    print(1)


#def plotFeat(model, fmapindex, imagepath):
def plotFeat(fmap, img, himg, receptivelayers, nodename, showintensity, f_min, f_max, prefix, canvas):
    #outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    #receptivelayers = mlutil.getConvLayerSubset(model, 'activation_3')
    #layermodel = mlutil.makeLayerOutputModel(model, outputlayers)
    #img = cv2.imread(imagepath)
    #img = cv2.resize(img, (256, 256))
    #outs = layermodel.predict(np.asarray([img]))
    #fmap = outs[-2][0, :, :, fmapindex]
    #f_min, f_max = fmap.min(), fmap.max()

    #h, himg = mlutil.heatmap(np.asarray([img]), model, 'activation_3', 0)
    #activationthreshold = 0.3
    #himg[himg < activationthreshold] = 0
    #himg[himg >= activationthreshold] = 1
    #heatimg = mlutil.overlayHeatmap2(np.asarray([img]), himg, 0.4)
    #cv2.imwrite('test_heat2.png', heatimg)

    # Scale image to 0-255
    if f_max == 0.0:
        scaled = fmap
    else:
        scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)

    # Create receptive field mask
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            if v > 0:
                xfield, yfield = mlutil.calcReceptiveField(x, y, receptivelayers)
                for xf in range(xfield[0], xfield[1] + 1):
                    for yf in range(yfield[0], yfield[1] + 1):
                        if mask[yf, xf][2] < v:
                            mask[yf, xf] = (0, 0, v if showintensity else 255)
                        #mask[yf, xf] = (0, 0, v)
    #himg = np.repeat(himg[:, :, np.newaxis], 3, axis=2)
    #mask2 = mask * himg
    #m = mask2.astype(np.uint8)
    #cv2.imwrite('test.png', mask2)

    alpha = 0.2
    #combined = cv2.addWeighted(img, 1.0, mask, 1.0, 0)
    combined = cv2.addWeighted(img, 0.4, mask, 1, 0)
    cv2.imwrite(prefix + 'test_' + nodename + '.png', combined)
    #if canvas is not None:
    #    img = canvas
    #combined = cv2.addWeighted(img, 1.0, m, 1.0, 0)
    #cv2.imwrite('test_full.png', combined)
    #return combined


def plotPat(model, pattern, prefix, imagepath):
    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    receptivelayers = mlutil.getConvLayerSubset(model, 'activation_3')
    layermodel = mlutil.makeLayerOutputModel(model, outputlayers)
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (256, 256))
    outs = layermodel.predict(np.asarray([img]))
    h, himg = mlutil.heatmap(np.asarray([img]), model, 'activation_3', 0)
    activationthreshold = 0.3
    himg[himg < activationthreshold] = 0
    himg[himg >= activationthreshold] = 1
    heatimg = mlutil.overlayHeatmap2(np.asarray([img]), himg, 0.4)
    cv2.imwrite(prefix + 'test_heat.png', heatimg)

    canvas = heatimg
    for p in pattern['pattern']:
        print(p)
        tokens = p.split('-')
        index = int(tokens[1])
        fmap = outs[-2][0, :, :, index]
        canvas = plotFeat(fmap, img, himg, receptivelayers, p, True, prefix + 'int_', canvas)
        #print(p)
    #cv2.imwrite(prefix + 'test_combined.png', canvas)

    canvas = None
    for p in pattern['pattern']:
        print(p)
        tokens = p.split('-')
        index = int(tokens[1])
        fmap = outs[-2][0, :, :, index]
        canvas = plotFeat(fmap, img, himg, receptivelayers, p, False, prefix, canvas)
        #print(p)
    cv2.imwrite(prefix + 'test_pattern.png', canvas)

    #for p in pattern['pattern']:
    #    print(p)
    #    tokens = p.split('-')
    #    index = int(tokens[1])
    #    fmap = outs[-2][0, :, :, index]
    #    plotFeat(fmap, img, himg, receptivelayers, p, True, prefix, None)
    #    #print(p)
    print(1)


def featuresToDataFrameNew(model, outputlayers, images):
    layermodel = mlutil.makeLayerOutputModel(model, outputlayers)

    pathindex = 0
    transactions = []
    for imagebatch, labelbatch in images:
        batchsize = imagebatch.shape[0]
        paths = images.file_paths[pathindex:pathindex + batchsize]
        pathindex += batchsize
        outs = layermodel.predict(imagebatch)
        for i in range(batchsize):
            trans = []
            pred = np.argmax(outs[-1][i])
            label = np.argmax(labelbatch[i])
            imagepath = paths[i]

            # Each layer
            li = 0
            for layer in outs[:-1]:
                li += 1

                # Each feature map in layer
                for fi in range(layer.shape[-1]):
                    print('image:' + str(pathindex + i) + ' layer:' + str(li - 1) + ' feat:' + str(fi))
                    v = layer[i, :, :, fi].max()
                    trans.append(v)

            trans.append(pred)
            trans.append(label)
            trans.append(imagepath)
            transactions.append(trans)

    # Generate dataframe
    head = []
    for l in outputlayers[:-1]:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    head.append('predicted')
    head.append('label')
    head.append('imagepath')
    df = pd.DataFrame(transactions, columns=head)
    df.index.name = 'index'
    return df


def plotLayer(model, maxvals, prefix, imagepath):
    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    receptivelayers = mlutil.getConvLayerSubset(model, 'activation_3')
    layermodel = mlutil.makeLayerOutputModel(model, outputlayers)
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (256, 256))
    outs = layermodel.predict(np.asarray([img]))

    o = outs[-2]
    for index in range(outs[-2].shape[-1]):
        print(index)
        #tokens = p.split('-')
        #index = int(tokens[1])
        fmap = outs[-2][0, :, :, index]
        mx = maxvals.loc['activation_3-' + str(index)]
        canvas = plotFeat(fmap, img, None, receptivelayers, 'activation_3-' + str(index), True, 0.0, mx, prefix, None)
        #print(p)
    print(1)


def plotPat2(model, pattern, maxvals, prefix, imagepath):
    outputlayers = ['activation', 'activation_1', 'prediction']
    receptivelayers = mlutil.getConvLayerSubset(model, 'activation_1')
    layermodel = mlutil.makeLayerOutputModel(model, outputlayers)
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (256, 256))
    outs = layermodel.predict(np.asarray([img]))

    o = outs[-2]
    #for index in range(outs[-2].shape[-1]):
    for p in pattern:
        print(p)
        tokens = p.split('-')
        index = int(tokens[1])
        mx = maxvals.loc[p]
        fmap = outs[-2][0, :, :, index]
        canvas = plotFeat(fmap, img, None, receptivelayers, 'activation_1-' + str(index), True, 0.0, mx, prefix, None)
        #print(p)
    print(1)


def geterrordataset(trans):
    minsup = 0.08
    minsupratio = 1.1
    strans = scaleframe(trans, 0, 255)
    binned = makebinary(strans, 128)
    newdf = pd.concat([binned, trans['predicted']], axis=1)
    col = [c for c in binned.columns if '_1' in c]
    sel = newdf.loc[newdf['label'] != newdf['predicted']].drop(['label', 'predicted'], axis=1)
    notsel = newdf.loc[newdf['label'] == newdf['predicted']].drop(['label', 'predicted'], axis=1)
    sel = sel.loc[:, col]
    notsel = notsel.loc[:, col]
    cpats = pats.mineContrastPatterns(sel, notsel, minsup, minsupratio)
    return cpats


trainds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)


#model = buildModel()
#model.fit(trainds, epochs=10, validation_data=valds)
#model.save('largeimage16.keras')


model = models.load_model('largeimage16.keras', compile=True)
#model = models.load_model('largeimage2.keras', compile=True)
outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
#trans = mlutil.featuresToDataFrame(model, outputlayers, valds, 0.3)
#trans = featuresToDataFrameNew(model, outputlayers, valds)
#trans.to_csv('session/trans_feat16.csv')


trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
trans = trans.loc[trans['label'] == trans['predicted']]
strans = scaleframe(trans, 0, 255)
binned = makebinary(strans, 128)
maxvals = trans.iloc[:, :-3].max()
col = [c for c in binned.columns if '_1' in c]
col.append('label')
binned = binned.loc[:, col]
binned.to_csv('activation_binned.csv')

minsup = 0.2
minsupratio = 2.0
#print('0')
#sel = binned.loc[binned['label'] == 0.0].drop('label', axis=1)
sel = binned.loc[binned['label'] == 0.0].drop('label', axis=1)
notsel = binned.loc[binned['label'] != 0.0].drop('label', axis=1)
cpats = pats.mineContrastPatterns(sel, notsel, minsup, minsupratio)
#cpats = geterrordataset(trans)

#test_loss, test_acc = model.evaluate(valds)
#w = model.layers[-1].get_weights()
#w[0][[52, 54, 55, 56, 59, 61], :] = [0, 0, 0]
#w[0][[20, 33, 40, 58], :] = [0, 0, 0]
#w[0][[20, 33, 40, 58], 0] = 0
#a = w[0][[20, 33, 40, 58], 0]
#w[0][20, 0] += 0.1
#w[0][33, 0] += 0.1
#model.layers[-1].set_weights(w)
#test_loss2, test_acc2 = model.evaluate(valds)
#d = frozenset(cpats[1]['targetmatches']) - frozenset(cpats[0]['targetmatches'])
#eqclass = list()
#eqclass.append([cpats[0], set(cpats[0]['targetmatches']), set(cpats[0]['othermatches'])])
#for p in cpats[1:]:
#    ts = set(p['targetmatches'])
#    os = set(p['othermatches'])
#    match = False
#    for eq in eqclass:
#        if ts == eq[1]: # and os == eq[2]:
#            match = True
#            break
#    if not match:
#        eqclass.append([p, ts, os])

#equivother = set()
#equivtarget = set()
#for p in cpats:
#    equivother.update(p['othermatches'])
#    equivtarget.update(p['targetmatches'])

#index = cpats[17]['targetmatches'][1]
#path = trans.loc[index, 'imagepath']
#plotPat(model, cpats[17], path)

#index = cpats[0]['targetmatches'][1]
#a = frozenset(sel.index.values)
#d = set(cpats[1]['targetmatches']) - (set(cpats[4]['targetmatches']).union(set(cpats[3]['targetmatches'])).union(set(cpats[2]['targetmatches'])))
#for index in d:
for index in cpats[1]['targetmatches']:
    print(index)
    pathprefix = 'session_pat/target/image_' + str(index) + '_'
    path = trans.loc[index, 'imagepath']
    plotPat2(model, cpats[1]['pattern'], maxvals, pathprefix, path)
    #plotLayer(model, maxvals, pathprefix, path)
print('1')
sel = binned.loc[binned['label'] == 1.0].drop('label', axis=1)
notsel = binned.loc[binned['label'] != 1.0].drop('label', axis=1)
cpats = pats.mineContrastPatterns(sel, notsel, minsup, minsupratio)
print('2')
sel = binned.loc[binned['label'] == 2.0].drop('label', axis=1)
notsel = binned.loc[binned['label'] != 2.0].drop('label', axis=1)
cpats = pats.mineContrastPatterns(sel, notsel, minsup, minsupratio)


#renderavg(trans)
#outs = strans.loc[strans['label'] == 2.0].iloc[0]
#outs = strans.iloc[38]
#renderactivations(strans.loc[strans['label'] == 0.0].iloc[0], 'active_0.png')
#renderactivations(strans.loc[strans['label'] == 1.0].iloc[0], 'active_1.png')
#renderactivations(strans.loc[strans['label'] == 2.0].iloc[0], 'active_2.png')


#s = trans.loc[trans['label'] == 0].iloc[:, :-3]
#mx = trans.iloc[:, :-3].max()
#mm = s.mean()
#for i in range(len(mm)):
#    mm[i] = scale(mm[i], 0.0, mx[i], 0, 255)
#    if math.isnan(mm[i]):
#        mm.iloc[i] = 0.0
#renderModel(model, mm)

print(1)
#g = nx.Graph()
#g.add_node(1)
#g.add_node(2)
#g.add_node(3)
#g.add_node(4)
#g.add_edge(1, 3)
#g.add_edge(1, 4)
#g.add_edge(2, 3)
#g.add_edge(2, 4)
#g.add_edge(1, 2)
#nx.draw(g)
#pos = graphviz_layout(g, prog='dot', args="-Grankdir=LR")
#nx.draw(g,with_labels=True,pos=pos, font_weight='bold')

#plt.show()

print(1)
