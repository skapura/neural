import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Functional
from datetime import datetime
from azure_data import load_dataset


def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
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


def train():
    trainds, valds = load_dataset('images_large')
    model = build_model()
    start = datetime.now()
    model.fit(trainds, validation_data=valds, epochs=2)
    l, a = model.evaluate(valds)
    print('loss: ' + str(l))
    print('accuracy: ' + str(a))
    end = datetime.now()
    print('training time: ' + str(end - start))
    model.save('trained.keras')


if __name__ == '__main__':
    train()
