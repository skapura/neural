from keras import models
import pandas as pd
#from data import load_dataset, load_dataset_selection, filter_dataset_paths, scale
import data
import patterns as pats
import mlutil
import plot
import azure as az
import const



def run():

    # Train model
    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    #model = az.train('src/models/golf.py')
    #model.save('src/models/golf.keras')
    model = models.load_model('src/models/golf.keras')
    patmodel = mlutil.make_output_model(model)

    # Load datasets
    trainds, valds = data.load_dataset('images_large')
    trainds = data.split_dataset_paths(trainds, 'golf_ball', label_mode='binary')
    valds = data.split_dataset_paths(valds, 'golf_ball', label_mode='binary')

    # Generate transactions
    trans = pats.model_to_transactions(patmodel, trainds)
    trans.to_csv('session/trans_binary_golf.csv')

    # Load transactions
    #trans = pd.read_csv('session/trans_binary_golf.csv', index_col='index')
    scaled = data.scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    # Render filters by layer
    franges = data.get_ranges(trans, zeromin=True)
    filters = patmodel.filters_in_layer('activation')
    path = data.image_path(trans, 0)
    plot.output_features(patmodel, path, filters, 'session/images', franges)

    # Restrict patterns to layer
    #col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation_3-' in c]

    sel = bdf.loc[bdf['label'] == 2.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 2.0].drop(const.META, axis=1)[col]
    minsup = 0.1
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)

    # dog patterns
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]
    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    print(1)