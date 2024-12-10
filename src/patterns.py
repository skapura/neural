import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from itertools import compress
import os
import spmf
import const
import data
import mlutil


def feature_activation_max(feature_map):
    if isinstance(feature_map, tf.Tensor):
        return tf.reduce_max(feature_map).numpy()
    else:
        return feature_map.max()


def features_match(features, pattern, scaler, feature_activation=feature_activation_max):
    matches = list()
    for f in features:
        pouts = [feature_activation(f[:, :, p]) for p in pattern]
        sv = data.scale_by_index(pouts, pattern, scaler)
        matches.append(all(s >= 0.5 for s in sv))
    return matches


def features_to_transactions(features, layer_name, feature_activation=feature_activation_max):
    transactions = []
    for finput in features:
        trans = []
        for fi in range(finput.shape[-1]):
            trans.append(feature_activation(finput[:, :, fi]))
        transactions.append(trans)

    head = [layer_name + '-' + str(i) for i in range(features.shape[-1])]
    df = pd.DataFrame(transactions, columns=head)
    df.index.name = 'index'
    return df


def model_to_transactions(model, ds, include_meta=True, feature_activation=feature_activation_max):
    batch_size = ds._input_dataset._batch_size.numpy()
    transactions = []
    for step, (x_batch, y_batch) in enumerate(ds):
        print('step: ' + str(step))
        #if step == 2:
        #    break
        idx = step * batch_size
        batch_paths = ds.file_paths[idx:idx + len(x_batch)]
        outs = model.predict(x_batch)

        # Iterate over each image in batch
        for i in range(len(x_batch)):
            if include_meta:
                if len(y_batch[i]) == 1:
                    pred = 1 if outs[-1][i][0] >= 0.5 else 0
                    label = y_batch[i].numpy()[0]
                else:
                    pred = np.argmax(outs[-1][i])
                    label = np.argmax(y_batch[i])

            # Collect outputs from each layer
            trans = []
            layerdata = outs[:-1] if isinstance(outs, list) else [outs]     # Convert to list if only 1 output
            for layer in layerdata:

                # Collect output from each filter in layer
                for fi in range(layer.shape[-1]):
                    trans.append(feature_activation(layer[i, :, :, fi]))
            if include_meta:
                trans += [pred, label, batch_paths[i]]
            transactions.append(trans)

    # Generate dataframe
    head = []
    outnames = model.output_names[:-1] if include_meta else model.output_names
    for l in outnames:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    if include_meta:
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


def matches(ds, pattern):
    imatch = []
    nonmatch = []
    for index, row in ds.iterrows():
        itemset = frozenset(compress(ds.columns, row))
        if pattern.issubset(itemset):
            imatch.append(index)
        else:
            nonmatch.append(index)
    return imatch, nonmatch


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
    targetmatches = set()
    othermatches = set()
    for index, p in enumerate(pats):
        targetmatches.update(p['targetmatches'])
        othermatches.update(p['othermatches'])
        for e in p['pattern']:
            if e in elems:
                elems[e]['patterns'].append(index)
                elems[e]['targetmatches'].update(p['targetmatches'])
                elems[e]['othermatches'].update(p['othermatches'])
            else:
                elems[e] = {'patterns': [index], 'targetmatches': set(p['targetmatches']), 'othermatches': set(p['othermatches'])}
    return elems, targetmatches, othermatches


def preprocess(base_model, ds, trans_path=None, scaler=None):

    # Load transactions
    if trans_path is None or not os.path.exists(trans_path):
        layermodel = mlutil.make_output_model(base_model)
        trans = model_to_transactions(layermodel, ds, include_meta=True)
        if trans is not None and trans_path is not None:
            trans.to_csv(trans_path)
    else:
        trans = pd.read_csv(trans_path, index_col='index')

    # Scale data
    if scaler is None:
        scaled, scaler = data.scale(trans, output_range=(0, 1))
    else:
        scaled, _ = data.scale(trans, output_range=(0, 1), scaler=scaler)

    bdf = binarize(scaled, 0.5)
    return bdf, scaler


def match_dataset(ds, bdf, pattern, pattern_class):
    classname = ds.class_names[pattern_class]
    matchidx, nonmatchidx = matches(bdf, set(pattern))
    matchds = data.load_dataset_selection(ds, bdf.loc[matchidx]['path'].to_list())
    patds = data.split_dataset_paths(matchds, classname, label_mode='binary')
    baseds = data.load_dataset_selection(ds, bdf.loc[nonmatchidx]['path'].to_list())
    return patds, baseds


def find_patterns(bdf, label, layer_name, min_sup=0.5, min_sup_ratio=1.1):
    sel, notsel = data.filter_transactions(bdf, layer_name + '-', label)
    cpats = mine_patterns(sel, notsel, min_sup, min_sup_ratio)
    return cpats
