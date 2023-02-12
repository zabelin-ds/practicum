import numpy as np
from tensorflow import keras

NP_LOAD_PATH = '/datasets/fashion_mnist/'

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')

    features_train = features_train.reshape(-1, 28 * 28) / 255.0

    return features_train, target_train


def create_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        units=10,
        input_dim=input_shape[0],
        activation='softmax'
    ))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=64, epochs=100,
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
