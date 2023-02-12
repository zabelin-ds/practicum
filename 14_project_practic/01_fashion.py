import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


# just 0.8631 of accuracy ..

optimizer = Adam(learning_rate=0.01)

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')

    features_train = features_train.reshape(-1, 28 * 28) / 255.0

    return features_train, target_train


def create_model(input_shape):
    model = keras.models.Sequential()
    # model.add(keras.layers.Dense(
    #     units=128,
    #     input_dim=input_shape[0],
    #     activation='relu'
    # ))
    model.add(keras.layers.Dense(
        units=64,
        # input_dim=input_shape[0],
        activation='relu'
    ))
    model.add(keras.layers.Dense(
        units=10,
        # input_dim=input_shape,
        activation='softmax'
    ))


    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    # adam
    # sgd

    print(model.summary())
    return model


def train_model(model, train_data, test_data, batch_size=128, epochs=57,
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

