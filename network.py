import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from keras import layers, models
import numpy as np
import mlutil
import cv2
import pandas as pd
import math
import patterns as pats


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
def plotFeat(fmap, img, himg, receptivelayers, nodename, showintensity, prefix, canvas):
    #outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    #receptivelayers = mlutil.getConvLayerSubset(model, 'activation_3')
    #layermodel = mlutil.makeLayerOutputModel(model, outputlayers)
    #img = cv2.imread(imagepath)
    #img = cv2.resize(img, (256, 256))
    #outs = layermodel.predict(np.asarray([img]))
    #fmap = outs[-2][0, :, :, fmapindex]
    f_min, f_max = fmap.min(), fmap.max()

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


def plotLayer(model, prefix, imagepath):
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

    #for p in pattern['pattern']:
    o = outs[-2]
    for index in range(outs[-2].shape[-1]):
        print(index)
        #tokens = p.split('-')
        #index = int(tokens[1])
        fmap = outs[-2][0, :, :, index]
        canvas = plotFeat(fmap, img, himg, receptivelayers, 'activation_3-' + str(index), True, prefix, None)
        #print(p)
    print(1)



valds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)

model = models.load_model('largeimage2.keras', compile=True)
outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
#trans = mlutil.featuresToDataFrame(model, outputlayers, valds, 0.3)
#trans.to_csv('session/trans_plot.csv')


trans = pd.read_csv('session/trans_plot.csv', index_col='index')
trans = trans.loc[trans['label'] == trans['predicted']]
strans = scaleframe(trans, 0, 255)
binned = makebinary(strans, 128)
col = [c for c in binned.columns if '_3' in c]
col.append('label')
binned = binned.loc[:, col]
binned.to_csv('activation_binned.csv')
#displaycounts(binned)
#inf = nodeig(binned, 'activation_3-0')
#infogain(binned)

minsup = 0.2
minsupratio = 2.0
print('0')
sel = binned.loc[binned['label'] == 0.0].drop('label', axis=1)
notsel = binned.loc[binned['label'] != 0.0].drop('label', axis=1)
cpats = pats.mineContrastPatterns(sel, notsel, minsup, minsupratio)

d = frozenset(cpats[1]['targetmatches']) - frozenset(cpats[0]['targetmatches'])
eqclass = list()
eqclass.append([cpats[0], set(cpats[0]['targetmatches']), set(cpats[0]['othermatches'])])
for p in cpats[1:]:
    ts = set(p['targetmatches'])
    os = set(p['othermatches'])
    match = False
    for eq in eqclass:
        if ts == eq[1]: # and os == eq[2]:
            match = True
            break
    if not match:
        eqclass.append([p, ts, os])

equivother = set()
equivtarget = set()
for p in cpats:
    equivother.update(p['othermatches'])
    equivtarget.update(p['targetmatches'])

#index = cpats[17]['targetmatches'][1]
#path = trans.loc[index, 'imagepath']
#plotPat(model, cpats[17], path)

index = cpats[0]['targetmatches'][1]
a = frozenset(sel.index.values)
d = a - frozenset(cpats[31]['targetmatches'])
for index in sel.index.values:
#for index in cpats[31]['othermatches']:
    print(index)
    pathprefix = 'session3/image_' + str(index) + '_'
    path = trans.loc[index, 'imagepath']
    #plotPat(model, cpats[31], pathprefix, path)
    plotLayer(model, pathprefix, path)
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
