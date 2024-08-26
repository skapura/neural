import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
import pandas as pd
from data import load_dataset, get_ranges, scale
import patterns as pats
import mlutil
import const

from keras.src.utils import dataset_utils
from keras.src.backend.config import standardize_data_format
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
import numpy as np


class PatternBranch(keras.Layer):
    def __init__(self):
        super().__init__()
        print('init pattern')

    def call(self, inputs):
        return inputs


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = PatternBranch()(x)
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
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def build_model_sel():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    #x = PatternBranch()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

def load_from_directory_test(
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
    selection=None
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

    if selection is not None:
        target = selection[0]
        other = selection[1]
        sel_image_paths = []
        sel_labels = []
        for path, label in zip(image_paths, labels):
            if path in target:
                sel_image_paths.append(path)
                sel_labels.append(0 if label_mode == 'binary' else label)
            elif path in other:
                sel_image_paths.append(path)
                sel_labels.append(1 if label_mode == 'binary' else label)
        image_paths = sel_image_paths
        labels = sel_labels
        if label_mode == 'binary':
            class_names = ['target', 'other']
        print(1)

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


def run():

    trainds, valds = load_dataset('images_large')

    #model = build_model()
    model = models.load_model('largeimage16.keras', compile=True)
    #w = oldmodel.get_weights()
    #model.set_weights(w)
    #test_loss, test_acc = model.evaluate(trainds)
    #test_loss_old, test_acc_old = oldmodel.evaluate(trainds)

    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    col = bdf.columns.difference(const.META, sort=False)
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]
    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    p = cpats[0]

    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    #vtrans = pats.model_to_transactions(model, outputlayers, valds)
    #vtrans.to_csv('session/vtrans_feat16.csv')
    vtrans = pd.read_csv('session/vtrans_feat16.csv', index_col='index')
    scaled = scale(vtrans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    imatch, _ = pats.matches(bdf.drop(const.META, axis=1), p['pattern'])
    labels = vtrans.iloc[imatch]['label']
    midx = list(labels[labels == 0].index.values)
    nidx = list(labels[labels != 0].index.values)
    t = vtrans.iloc[midx]['path'].to_list()
    o = vtrans.iloc[nidx]['path'].to_list()
    valds_sel = load_from_directory_test('datasets/' + 'images_large' + '/val', labels='inferred', label_mode='binary',
                                  image_size=(256, 256), shuffle=True, selection=(t, o))
    valds_sel2 = load_from_directory_test('datasets/' + 'images_large' + '/val', labels='inferred', label_mode='categorical',
                                         image_size=(256, 256), shuffle=True, selection=(t, o))
    selmodel = models.load_model('submodel.keras')
    model = models.load_model('largeimage16.keras')
    loss_sel, acc_sel = selmodel.evaluate(valds_sel)
    ls, acc = model.evaluate(valds_sel2)

    t = trans.iloc[p['targetmatches']]['path'].to_list()
    o = trans.iloc[p['othermatches']]['path'].to_list()
    trainds_sel = load_from_directory_test('datasets/' + 'images_large' + '/train', labels='inferred', label_mode='binary',
                                  image_size=(256, 256), shuffle=True, selection=(t, o))
    trainds2 = load_from_directory_test('datasets/' + 'images_large' + '/train', labels='inferred',
                                           label_mode='categorical',
                                           image_size=(256, 256), shuffle=True, selection=(t, o))
    #acc_sel, loss_sel = selmodel.evaluate(trainds_sel)
    #acc, ls = model.evaluate(trainds2)

    submodel = build_model_sel()
    submodel.fit(trainds_sel, validation_data=valds_sel, epochs=10)
    submodel.save('submodel.keras')

    print(1)
