import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def build_model(rows, columns, ship_sizes):
    """
    Build a simple CNN model to predict ship location density from observed
    hits and misses
    """

    input_layer = keras.Input(shape=(rows, columns, 1))

    # note that we're using padding=same to preserve the shape since
    # our eventual output should have the same shape as the input
    conv_1 = layers.Conv2D(
        16, kernel_size=5, strides=1, use_bias=True, activation="relu",
        padding="same",
    )(input_layer)
    batch_1 = layers.BatchNormalization()(conv_1)

    conv_2 = layers.Conv2D(
        8, kernel_size=3, strides=1, activation="relu", padding="same"
    )(batch_1)
    batch_2 = layers.BatchNormalization()(conv_2)

    # use a sigmoid activation on the last layer because we know all the values
    # in our target arrays are in [0, 1]
    conv_3 = layers.Conv2D(
        4, kernel_size=3, strides=1, activation="sigmoid", padding="same"
    )(batch_2)

    flattened = tf.math.reduce_mean(conv_3, axis=-1, keepdims=True)

    model = keras.Model(inputs=[input_layer], outputs=[flattened])

    # use the pixel-wise mean squared error for now
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")

    return model
