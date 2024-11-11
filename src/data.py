from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.src.utils import dataset_utils
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
from keras.src.backend.config import standardize_data_format
import pandas as pd
import numpy as np
import pickle
import math
import const


ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def load_from_directory(
    directory,
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=None,
    sample_size=None
):

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        'inferred',
        formats=ALLOWLIST_FORMATS,
        class_names=None,
        shuffle=shuffle,
        seed=seed,
        follow_links=False
    )

    if not image_paths:
        raise ValueError(
            f"No images found in directory {directory}. "
            f"Allowed formats: {ALLOWLIST_FORMATS}"
        )

    if sample_size is not None:
        samples = np.random.default_rng().choice(len(image_paths), sample_size, replace=False)
        image_paths = [image_paths[s] for s in samples]
        labels = [labels[s] for s in samples]

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=3,
        labels=labels,
        label_mode='categorical',
        num_classes=len(class_names) if class_names else 0,
        interpolation='bilinear',
        crop_to_aspect_ratio=False,
        data_format='channels_last',
    )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset.class_names = class_names
    dataset.file_paths = image_paths
    dataset.labels = list(labels)

    return dataset


def labels_to_binary(labels, target):
    return [1 if lbl == target else 0 for lbl in labels]


def image_subset(image_paths, labels, class_names, binary_target=None, batch_size=32, image_size=(256, 256)):

    if binary_target is not None:
        labels = labels_to_binary(labels, binary_target)
        class_names = ['other', 'target']

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=3,
        labels=labels,
        label_mode='categorical' if binary_target is None else 'binary',
        num_classes=len(class_names) if class_names else 0,
        interpolation='bilinear',
        crop_to_aspect_ratio=False,
        data_format='channels_last',
    )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset.class_names = class_names
    dataset.file_paths = image_paths
    dataset.labels = labels

    return dataset


def filter_transactions(df, column_prefix, label):
    col = [c for c in df.columns if column_prefix in c]
    sel = df.loc[df['label'] == label].drop(const.META, axis=1)[col]
    notsel = df.loc[df['label'] != label].drop(const.META, axis=1)[col]
    print('target # instances: ' + str(len(sel)))
    print('other # instances: ' + str(len(notsel)))
    return sel, notsel



####################################

# modified keras image_dataset_from_directory
def load_from_directory2(
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
    sample_size=None,
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

    # Subsampling
    if sample_size is not None:
        samples = np.random.default_rng().choice(len(image_paths), sample_size, replace=False)
        image_paths = [image_paths[s] for s in samples]
        labels = [labels[s] for s in samples]

    # Added: subset selection and binary translation
    if selection is not None:
        sel_image_paths = []
        sel_labels = []

        # Divide into target-other subsets
        if isinstance(selection, tuple):
            target = selection[0]
            other = selection[1]

            class_names = ['other', 'target']
            for path, label in zip(image_paths, labels):
                if path in target:
                    sel_image_paths.append(path)
                    sel_labels.append(1)
                elif path in other:
                    sel_image_paths.append(path)
                    sel_labels.append(0)

        # Generic image selection
        else:
            for path, label in zip(image_paths, labels):
                if path in selection:
                    sel_image_paths.append(path)
                    sel_labels.append(label)

        image_paths = sel_image_paths
        labels = sel_labels

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


def load_dataset(dsname, size=(256, 256), shuffle=True, sample_size=None):
    trainds = load_from_directory('datasets/' + dsname + '/train', labels='inferred', label_mode='categorical', image_size=size, shuffle=shuffle, sample_size=sample_size)
    valds = load_from_directory('datasets/' + dsname + '/val', labels='inferred', label_mode='categorical', image_size=size, shuffle=shuffle, sample_size=sample_size)
    return trainds, valds


def filter_dataset_paths(ds, contains_text, in_flag=True):
    if in_flag:
        selected = [r for r in ds.file_paths if contains_text in r]
    else:
        selected = [r for r in ds.file_paths if contains_text not in r]
    return selected


def split_dataset_paths(ds, contains_text, label_mode='categorical'):
    matches = filter_dataset_paths(ds, contains_text, in_flag=True)
    notmatches = filter_dataset_paths(ds, contains_text, in_flag=False)
    splitds = load_dataset_selection(ds, selection=(matches, notmatches), label_mode=label_mode)
    return splitds


def load_dataset_selection(ds, selection, label_mode='categorical', size=(256, 256), shuffle=True):
    if not isinstance(ds, str):
        splits = ds.file_paths[0].split('/')[:-2]
        dspath = '/'.join(splits)
    else:
        dspath = 'datasets/' + ds

    selds = load_from_directory(dspath, labels='inferred', label_mode=label_mode, image_size=size, shuffle=shuffle, selection=selection)
    return selds


def scale(df, output_range=(0, 1), scaler=None, exclude=const.META):
    selected = df.columns.difference(exclude, sort=False)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=output_range)
        scaler.fit(df[selected])
    #scaled = scaler.fit_transform(df[selected])
    scaled = scaler.transform(df[selected])
    sdf = pd.DataFrame(scaled, columns=selected, index=df.index)
    #a = sdf.astype(int)
    sdf = pd.concat([sdf, df[exclude]], axis=1)
    return sdf, scaler


def scale_by_index(vals, index, scaler):
    s = [scaler.scale_[i] for i in index]
    m = [scaler.min_[i] for i in index]
    scaled = [v * sv + mv for v, sv, mv in zip(vals, s, m)]
    return scaled


def get_ranges(df, zeromin=False, exclude=const.META):
    selected = df.columns.difference(exclude, sort=False)
    sdf = pd.DataFrame(df, columns=selected, index=df.index)
    minvals = sdf.min()
    maxvals = sdf.max()
    r = {}
    for col in selected:
        r[col] = (0.0 if zeromin else minvals.loc[col], maxvals.loc[col])
    return r


def image_path(trans, index):
    return trans.loc[index]['path']


def entropy(pvals):
    info = 0.0
    for p in pvals:
        if p > 0.0:
            info -= p * math.log2(p)
    return info


def info_gain(cprobs, asup, aprobs):
    if asup == 0.0:
        return 0.0
    nasup = 1.0 - asup
    naprobs = [1.0 - p for p in aprobs]
    infoa = asup * entropy(aprobs) + nasup * entropy(naprobs)
    return entropy(cprobs) - infoa


def activation_info(bdf, exclude=const.META):

    # Class statistics
    classinfo = {'counts': {}, 'support': {}}
    classes = bdf['label'].unique()
    classes.sort()
    for cls in classes:
        classcount = len(bdf[bdf['label'] == cls])
        classinfo['counts'][cls] = classcount
        classinfo['support'][cls] = classcount / len(bdf)
    classinfo['info'] = entropy(classinfo['support'].values())

    col = bdf.columns.difference(exclude, sort=False)
    stats = []
    for c in col:
        total = 0
        row = []
        counts = []
        supports = []
        for cls in classes:
            classcount = len(bdf[(bdf[c] == 1) & (bdf['label'] == cls)])
            counts.append(classcount)
            supports.append(classcount / classinfo['counts'][cls])
            total += classcount
        row += counts
        row += supports
        row.append(total)
        gsupport = total / len(bdf)
        row.append(gsupport)
        row.append(info_gain(classinfo['support'].values(), gsupport, supports))
        stats.append(row)

    head = []
    for cls in classes:
        head.append('count-' + str(cls))
    for cls in classes:
        head.append('support-' + str(cls))
    head += ['gcount', 'gsupport', 'infogain']
    adf = pd.DataFrame(stats, columns=head, index=col)
    return classinfo, adf

def save(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
