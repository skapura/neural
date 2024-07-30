import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from keras import layers, models
import numpy as np
import mlutil
import cv2
import pandas as pd
import math


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


def renderactivations(outs):
    print(1)


valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)

model = models.load_model('largeimage2.keras', compile=True)
outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
#trans = mlutil.featuresToDataFrame(model, outputlayers, valds)
#trans.to_csv('session/trans_plot.csv')
trans = pd.read_csv('session/trans_plot.csv', index_col='index')
trans = trans.loc[trans['label'] == trans['predicted']]
strans = scaleframe(trans, 0, 255)
renderavg(trans)
outs = strans.iloc[0]
renderactivations(outs)


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
