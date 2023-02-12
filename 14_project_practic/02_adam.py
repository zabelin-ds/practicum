import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, AveragePooling2D, MaxPooling2D


optimizer = Adam(learning_rate=0.0027)

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')

    # features_train = features_train / 255.0
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0

    return features_train, target_train


def create_model(input_shape):
    print(input_shape)

    model = keras.models.Sequential()

    model.add(Conv2D(
        input_shape=input_shape,
        filters=17,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
    ))

    model.add(AveragePooling2D(
        pool_size=(2, 2),
        strides=None,
        padding='valid'
    ))

    model.add(Conv2D(
        filters=7,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        strides=2
    ))

    model.add(Flatten())

    model.add(keras.layers.Dense(
        units=10,
        activation='softmax'
    ))


    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    # adam

    print(model.summary())
    return model


def train_model(model, train_data, test_data, batch_size=144, epochs=77,
                steps_per_epoch=None,
                validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train,
            validation_data=(features_test, target_test),
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
    )

    return model

