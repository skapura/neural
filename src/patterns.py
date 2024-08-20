import mlutil
import pandas as pd
import numpy as np


def feature_activation_max(feature_map):
    return feature_map.max()


def model_to_transactions(model, layers, ds, feature_activation=feature_activation_max):
    layermodel = mlutil.make_output_model(model, layers)

    batch_size = ds._input_dataset._batch_size.numpy()
    transactions = []
    for step, (x_batch, y_batch) in enumerate(ds):
        print('step: ' + str(step))
        if step == 2:
            break
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
    head += ['predicted', 'label', 'path']
    df = pd.DataFrame(transactions, columns=head)
    df.index.name = 'index'
    return df


def binarize(df, threshold=0.5, exclude=['predicted', 'label', 'path']):
    selected = df.columns.difference(exclude, sort=False)
    binned = df[selected].map(lambda x: 1 if x >= threshold else 0)
    binned = pd.concat([binned, df[exclude]], axis=1)
    return binned
