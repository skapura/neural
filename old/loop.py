import tensorflow as tf
import keras
from keras.src.models import Functional
from keras import layers
import numpy as np
import mlutil


class PatternModel(keras.Model):

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}


def buildModel():
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
    #x = layers.Dense(3, name='prediction')(x)
    model = PatternModel(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


class PatternModel(keras.Model):

    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

@tf.function
def trainstep(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch_train, logits)
    return loss_value


trainds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)

model = buildModel()
model.fit(trainds, validation_data=valds, epochs=2)
test_loss, test_acc = model.evaluate(valds)
#optimizer = keras.optimizers.SGD(learning_rate=1e-3)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_metric = keras.metrics.CategoricalAccuracy()

epochs = 2
for e in range(epochs):
    print('epoch:' + str(e))
    for step, (x_batch_train, y_batch_train) in enumerate(trainds):
        print('step:' + str(step))
        loss_value = trainstep(x_batch_train, y_batch_train)


        #if step % 200 == 0:
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step, float(loss_value))
        )
        train_acc = train_acc_metric.result()
        print('accuracy: ' + str(float(train_acc)))
        print("Seen so far: %s samples" % ((step + 1) * 32))
