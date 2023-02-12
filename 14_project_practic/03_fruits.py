import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50

optimizer = Adam(learning_rate=0.0027)
datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1.0 / 255
)


def load_train(path):
    features_train = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        # указываем, что это загрузчик для обучающей выборки
        subset='training',
        seed=108108108
    )

    return features_train


def create_model(input_shape):
    print(input_shape)

    model = keras.models.Sequential()

    model.add(Conv2D(
        input_shape=input_shape,
        filters=17,
        kernel_size=(4, 4),
        activation='relu',
        padding='same',
    ))

    model.add(AveragePooling2D(
        pool_size=(2, 2),
        strides=None,
        padding='valid'
    ))

    model.add(Conv2D(
        filters=34,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        strides=2
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=None,
        padding='valid'
    ))

    model.add(Flatten())

    model.add(keras.layers.Dense(
        units=12,
        activation='softmax'
    ))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )

    print(model.summary())
    return model


def train_model(model,
                features_train,
                features_test,
                batch_size=None,
                epochs=3,
                steps_per_epoch=None,
                validation_steps=None
                ):
    model.fit(features_train,
              validation_data=features_test,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_steps=validation_steps,
              steps_per_epoch=steps_per_epoch,
              # shuffle=True,
              )

    return model
