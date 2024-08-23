import pandas as pd
import numpy as np
import csv
from itertools import compress
import os
import spmf
import const
import mlutil


def feature_activation_max(feature_map):
    return feature_map.max()


def model_to_transactions(model, layers, ds, feature_activation=feature_activation_max):
    layermodel = mlutil.make_output_model(model, layers)

    batch_size = ds._input_dataset._batch_size.numpy()
    transactions = []
    for step, (x_batch, y_batch) in enumerate(ds):
        print('step: ' + str(step))
        #if step == 2:
        #    break
        idx = step * batch_size
        batch_paths = ds.file_paths[idx:idx + len(x_batch)]
        outs = layermodel.predict(x_batch)

        # Iterate over each image in batch
        for i in range(len(x_batch)):
            pred = np.argmax(outs[-1][i])
            label = np.argmax(y_batch[i])

            # Collect outputs from each layer
            trans = []
            for layer in outs[:-1]:

                # Collect output from each filter in layer
                for fi in range(layer.shape[-1]):
                    trans.append(feature_activation(layer[i, :, :, fi]))
            trans += [pred, label, batch_paths[i]]
            transactions.append(trans)

    # Generate dataframe
    head = []
    for l in layers[:-1]:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    head += const.META
    df = pd.DataFrame(transactions, columns=head)
    df.index.name = 'index'
    return df


def binarize(df, threshold=0.5, exclude=const.META):
    selected = df.columns.difference(exclude, sort=False)
    binned = df[selected].map(lambda x: 1 if x >= threshold else 0)
    binned = pd.concat([binned, df[exclude]], axis=1)
    return binned


def mine_patterns(target, other, minsup, supratio, maxlen=5):

    # Write transactions to temp file
    lookup = {}
    for i in range(len(target.columns)):
        lookup[target.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in target.iterrows():
            itemset = [lookup[item] for item in list(compress(target.columns, row))]
            writer.writerow(itemset)

    # Perform pattern mining on 'target' dataset
    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[minsup, maxlen])
    fpclose.run()
    fpclose.parse_output()
    os.remove('.trans.csv')
    os.remove('.patterns.csv')
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([target.columns[int(itm)] for itm in items[:-2]])

        # TODO: there is a bug in the way SPFM was modified to restict pattern length
        if len(pat) <= maxlen:
            sup = int(items[-1]) / float(len(target))
            patlist.append({'pattern': pat, 'targetsupport': sup, 'othermatches': [], 'targetmatches': []})

    # Convert target/other datasets to sets for pattern matching
    othersets = []
    for index, row in other.iterrows():
        itemset = frozenset(compress(other.columns, row))
        othersets.append({'index': index, 'itemset': itemset})
    targetsets = []
    for index, row in target.iterrows():
        itemset = frozenset(compress(target.columns, row))
        targetsets.append({'index': index, 'itemset': itemset})

    # Find pattern support in both datasets
    for p in patlist:
        count = 0
        for c in othersets:
            if p['pattern'].issubset(c['itemset']):
                p['othermatches'].append(c['index'])
                count += 1
        for c in targetsets:
            if p['pattern'].issubset(c['itemset']):
                p['targetmatches'].append(c['index'])
        p['othersupport'] = count / float(len(othersets))
        p['supportdiff'] = p['targetsupport'] - p['othersupport']
        p['supportratio'] = p['targetsupport'] / p['othersupport'] if p['othersupport'] > 0.0 else 9999.0

    # Prune low support/contrast patterns
    patlist.sort(key=lambda x: x['supportratio'], reverse=True)
    selectedpats = []
    for p in patlist:
        if p['supportratio'] >= supratio and p['targetsupport'] >= minsup:
            selectedpats.append(p)
    for p in reversed(selectedpats):
        print(str(p['targetsupport'] - p['othersupport']) + ', ' + str(p['othersupport']) + ', ' + str(p['targetsupport']) + ', ' + str(p['supportratio']) + ', ' + str(p['pattern']))
    return selectedpats


def filter_patterns_by_layer(pats, layers):
    matches = []
    for p in pats:
        for elem in p['pattern']:
            if any(l in elem for l in layers):
                matches.append(p)
                break
    return matches


def unique_elements(pats):
    elems = {}
    for index, p in enumerate(pats):
        for e in p['pattern']:
            if e in elems:
                elems[e]['patterns'].append(index)
                elems[e]['targetmatches'].union(p['targetmatches'])
                elems[e]['othermatches'].union(p['othermatches'])
            else:
                elems[e] = {'patterns': [index], 'targetmatches': set(p['targetmatches']), 'othermatches': set(p['othermatches'])}
    return elems
