import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
from datetime import datetime
from azure_data import load_dataset, filter_dataset_paths, load_dataset_selection


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (22, 22))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    #x = layers.Conv2D(128, (3, 3))(x)
    #x = layers.Activation('relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def train():
    trainds, valds = load_dataset('images_large')

    golfimages = filter_dataset_paths(trainds, 'golf_ball')
    notgolfimages = filter_dataset_paths(trainds, 'golf_ball', in_flag=False)
    golftrainds = load_dataset_selection('images_large', selection=(golfimages, notgolfimages), label_mode='binary')

    golfimages = filter_dataset_paths(valds, 'golf_ball')
    notgolfimages = filter_dataset_paths(valds, 'golf_ball', in_flag=False)
    golfvalds = load_dataset_selection('images_large', selection=(golfimages, notgolfimages), label_mode='binary')

    model = build_model()
    start = datetime.now()
    model.fit(golftrainds, validation_data=golfvalds, epochs=100)
    l, a = model.evaluate(valds)
    print('loss: ' + str(l))
    print('accuracy: ' + str(a))
    end = datetime.now()
    print('training time: ' + str(end - start))
    model.save('trained.keras')


if __name__ == '__main__':
    train()
