import numpy as np
from tensorflow.keras import models, callbacks
from keras.src.utils import dataset_utils
from keras.src.utils import image_utils
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
from keras.src.backend.config import standardize_data_format
import tensorflow as tf
import cv2
import pandas as pd
import shutil
import os
import glob

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def makeDebugModel(model, onlyconv=False):
    if onlyconv:
        outputs = [layer.output for layer in model.layers if type(layer) is keras.layers.Conv2D]
    else:
        outputs = [layer.output for layer in model.layers]
    debugmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return debugmodel


def makeLayerOutputModel(model, layernames):
    outputs = [layer.output for layer in model.layers if layer.name in layernames]
    outputmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return outputmodel


def getConvInfo(model):
    convinfo = [{'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides}
                for l in model.layers if 'conv2d' in l.name or 'max_pooling2d' in l.name]
    return convinfo


def layerIndex(model, layername):
    for li in range((len(model.layers))):
        if model.layers[li].name == layername:
            return li
    return None


def calcReceptiveField(x, y, layers):
    startx = x
    endx = x
    starty = y
    endy = y
    for l in reversed(layers):
        startx = startx * l['stride'][1]
        endx = endx * l['stride'][1] + l['kernel'][1] - 1
        starty = starty * l['stride'][0]
        endy = endy * l['stride'][0] + l['kernel'][0] - 1
    return (startx, endx), (starty, endy)


def heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output[0]]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
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
    heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmapimage = cv2.resize(heatmap, (w, h))

    return heatmap, heatmapimage


def overlayHeatmap(image, heat, alpha=0.4):
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(heat, (w, h))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 0
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    colormap = cv2.COLORMAP_VIRIDIS
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image[0], alpha, heatmap, 1 - alpha, 0)
    return output


def getLayerOutputRange(model, layernames, images, franges=None):
    layermodel = makeLayerOutputModel(model, layernames)

    class LayerCallback(callbacks.Callback):
        ranges = []

        def __init__(self, model, franges):
            if franges is None:
                for l in model.output_shape:
                    self.ranges.append([(9999.0, -9999.0) for _ in range(l[-1])])
            else:
                self.ranges = franges
            super(callbacks.Callback, self).__init__()


        def on_predict_batch_end(self, batch, logs=None):
            outs = logs['outputs']
            for i in range(0, len(outs)):
                for fi in range(outs[i].shape[-1]):
                    fmaps = outs[i][:, :, :, fi] if len(outs[i].shape) == 4 else outs[i]
                    minval = tf.reduce_min(fmaps).numpy()
                    maxval = tf.reduce_max(fmaps).numpy()
                    c = self.ranges[i][fi]
                    self.ranges[i][fi] = (min(c[0], minval), max(c[1], maxval))

    layerinfo = LayerCallback(layermodel, franges)
    layermodel.predict(images, callbacks=[layerinfo])
    return layerinfo.ranges


def getConvLayerSubset(model, lastlayer):
    convinfo = getConvInfo(model)
    idx = layerIndex(model, lastlayer)
    convname = model.layers[idx - 1].name
    receptivelayers = []
    for l in convinfo:
        receptivelayers.append(l)
        if l['name'] == convname:
            break
    return receptivelayers


def combineHeatAndFeatures(model, franges, outputlayers, lastconvlayer, image):
    # Get initial outputs
    layermodel = makeLayerOutputModel(model, outputlayers)
    outs = layermodel.predict(np.asarray([image]))
    #outs = layermodel(np.asarray([image]), training=False)
    predicted = np.argmax(outs[-1])
    _, himg = heatmap(np.asarray([image]), model, lastconvlayer, predicted)

    himg[himg < .7] = 0
    himg[himg >= .7] = 1

    heats = []
    for layerindex in range(len(outputlayers) - 1):

        # Get subset of conv layers
        receptivelayerinfo = getConvLayerSubset(model, outputlayers[layerindex])

        # layerindex = 1
        fmaps = outs[layerindex][0]
        r = franges[layerindex]
        currlayerheats = []
        for fi in range(fmaps.shape[2]):
            if r[fi][1] == 0.0:
                currfmap = fmaps[:, : fi]
            else:
                currfmap = (fmaps[:, :, fi] - r[fi][0]) / (r[fi][1] - r[fi][0])
            heatfeatmap = np.zeros(currfmap.shape)
            for x in range(currfmap.shape[1]):
                for y in range(currfmap.shape[0]):
                    xrange, yrange = calcReceptiveField(x, y, receptivelayerinfo)
                    v = currfmap[y, x]

                    # Use all vals in receptive field on heatmap
                    hr = himg[yrange[0]:yrange[1] + 1, xrange[0]:xrange[1] + 1]
                    heatfeatmap[y, x] = v * np.sum(hr)

                    # Only use center val in receptive field on heatmap (assume 3x3 kernel)
                    # heatfeatmap[y, x] = v * himg[yrange[0] + 1, xrange[0] + 1]
            currlayerheats.append(heatfeatmap)
        heats.append(currlayerheats)
    return heats, predicted


def featuresToDataFrame(model, outputlayers, images):
    maskheat = False

    layermodel = makeLayerOutputModel(model, outputlayers)

    # Generate header row
    head = []
    for l in outputlayers[:-1]:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    head.append('predicted')
    head.append('label')
    head.append('imagepath')

    #if not os.path.exists('.session'):
    #    os.makedirs('.session')
    #else:
    #    for f in glob.glob('heat_*'):
    #        os.remove(f)

    # Convert each list of feature map activations to transactions
    transactions = []
    pathindex = 0
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
            if maskheat:
                _, himg = heatmap(np.asarray([imagebatch[i, :, :, :]]), model, 'activation_3', pred)
            #np.save('heat_' + str(pathindex).zfill(5) + '.npy', himg)
            li = 0
            for layer in outs[:-1]:

                receptivelayerinfo = getConvLayerSubset(model, outputlayers[li])
                li += 1

                for fi in range(layer.shape[-1]):
                    #print('image:' + str(pathindex + i) + ' layer:' + str(li - 1) + ' feat:' + str(fi))

                    if maskheat:
                        currfmap = layer[i, :, :, fi]
                        heatfeatmap = np.zeros(currfmap.shape)
                        for x in range(currfmap.shape[1]):
                            for y in range(currfmap.shape[0]):
                                xrange, yrange = calcReceptiveField(x, y, receptivelayerinfo)
                                v = currfmap[y, x]
                                hr = himg[yrange[0]:yrange[1] + 1, xrange[0]:xrange[1] + 1]
                                heatfeatmap[y, x] = v * np.sum(hr)

                        v = heatfeatmap.max()
                    else:
                        v = layer[i, :, :, fi].max()
                    trans.append(v)
            trans.append(pred)
            trans.append(label)
            trans.append(imagepath)
            transactions.append(trans)

    df = pd.DataFrame(transactions, columns=head)
    #df.set_index('index', inplace=True)
    df.index.name = 'index'
    return df


def featuresToDataFrame2(model, outputlayers, lastlayer, franges, indexes, images, labels):

    # Generate header row
    head = ['index']
    for l in outputlayers[:-1]:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    head.append('predicted')
    head.append('label')

    # Convert each list of feature map activations to transactions
    transactions = []
    for i in range(len(images)):
        print(str(i) + '/' + str(len(images)))
        index = indexes[i]
        img = images[i]
        label = labels[i]
        heats, predicted = combineHeatAndFeatures(model, franges, outputlayers, lastlayer, img)

        trans = [index]
        for layerindex in range(len(heats)):
            for fi in range(len(heats[layerindex])):
                v = heats[layerindex][fi].max()
                trans.append(v)
        trans.append(predicted)
        trans.append(label)
        transactions.append(trans)

    df = pd.DataFrame(transactions, columns=head)
    df.set_index('index', inplace=True)
    return df

# modified keras image_dataset_from_directory
def load_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
):
    """Generates a `tf.data.Dataset` from image files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataset_from_directory(main_directory,
    labels='inferred')` will return a `tf.data.Dataset` that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.
    Animated gifs are truncated to the first frame.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing images for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            image files found in the directory. Labels should be sorted
            according to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `labels` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`.
            Defaults to `"rgb"`. Whether the images will be converted to
            have 1, 3, or 4 channels.
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        image_size: Size to resize images to after they are read from disk,
            specified as `(height, width)`. Defaults to `(256, 256)`.
            Since the pipeline processes batches of images that must all have
            the same size, this must be provided.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"`, or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        interpolation: String, the interpolation method used when
            resizing images. Defaults to `"bilinear"`.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
        follow_links: Whether to visit subdirectories pointed to by symlinks.
            Defaults to `False`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the
            target aspect ratio, the output image will be cropped so as to
            return the largest possible window in the image
            (of size `image_size`) that matches the target aspect ratio. By
            default (`crop_to_aspect_ratio=False`), aspect ratio may not be
            preserved.
        data_format: If None uses keras.config.image_data_format()
            otherwise either 'channel_last' or 'channel_first'.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
    - Otherwise, it yields a tuple `(images, labels)`, where `images` has
        shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

    Rules regarding labels format:

    - if `label_mode` is `"int"`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `"binary"`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `"categorical"`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

    - if `color_mode` is `"grayscale"`,
        there's 1 channel in the image tensors.
    - if `color_mode` is `"rgb"`,
        there are 3 channels in the image tensors.
    - if `color_mode` is `"rgba"`,
        there are 4 channels in the image tensors.
    """

    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "`labels` argument should be a list/tuple of integer labels, "
                "of the same size as the number of image files in the target "
                "directory. If you wish to infer the labels from the "
                "subdirectory "
                'names in the target directory, pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains images "
                f"(no labels), pass `labels=None`. Received: labels={labels}"
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if "
                f'`labels="inferred"`. Received: labels={labels}, and '
                f"class_names={class_names}"
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", '
            '"categorical", "binary", '
            f"or None. Received: label_mode={label_mode}"
        )
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )

    interpolation = interpolation.lower()
    supported_interpolations = (
        "bilinear",
        "nearest",
        "bicubic",
        "area",
        "lanczos3",
        "lanczos5",
        "gaussian",
        "mitchellcubic",
    )
    if interpolation not in supported_interpolations:
        raise ValueError(
            "Argument `interpolation` should be one of "
            f"{supported_interpolations}. "
            f"Received: interpolation={interpolation}"
        )

    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=ALLOWLIST_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary"`, there must be exactly 2 '
            f"class_names. Received: class_names={class_names}"
        )

    data_format = standardize_data_format(data_format=data_format)

    if subset == "both":
        (
            image_paths_train,
            labels_train,
        ) = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, "training"
        )
        (
            image_paths_val,
            labels_val,
        ) = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, "validation"
        )
        if not image_paths_train:
            raise ValueError(
                f"No training images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )
        if not image_paths_val:
            raise ValueError(
                f"No validation images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )
        train_dataset = paths_and_labels_to_dataset(
            image_paths=image_paths_train,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels_train,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            data_format=data_format,
        )

        val_dataset = paths_and_labels_to_dataset(
            image_paths=image_paths_val,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels_val,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            data_format=data_format,
        )

        if batch_size is not None:
            if shuffle:
                # Shuffle locally at each iteration
                train_dataset = train_dataset.shuffle(
                    buffer_size=batch_size * 8, seed=seed
                )
            train_dataset = train_dataset.batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)
        else:
            if shuffle:
                train_dataset = train_dataset.shuffle(
                    buffer_size=1024, seed=seed
                )

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # Users may need to reference `class_names`.
        train_dataset.class_names = class_names
        val_dataset.class_names = class_names

        # Include file paths for images as attribute.
        train_dataset.file_paths = image_paths_train
        val_dataset.file_paths = image_paths_val

        dataset = [train_dataset, val_dataset]
    else:
        image_paths, labels = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, subset
        )
        if not image_paths:
            raise ValueError(
                f"No images found in directory {directory}. "
                f"Allowed formats: {ALLOWLIST_FORMATS}"
            )

        dataset = paths_and_labels_to_dataset(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=len(class_names) if class_names else 0,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            data_format=data_format,
        )

        if batch_size is not None:
            #if shuffle:
            #    # Shuffle locally at each iteration
            #    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
            dataset = dataset.batch(batch_size)
        #else:
        #    if shuffle:
        #        dataset = dataset.shuffle(buffer_size=1024, seed=seed)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        # Users may need to reference `class_names`.
        dataset.class_names = class_names

        # Include file paths for images as attribute.
        dataset.file_paths = image_paths

    return dataset