# import os
import pandas as pd

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50


IMG_SIDE = 256
TO_IMG_SIZE = (IMG_SIDE, IMG_SIDE)
SEED = 108108108
COLOR_MODE = 'rgb'
# COLOR_MODE = 'grayscale'

optimizer = Adam(learning_rate=0.0005)
datagen = ImageDataGenerator(
    validation_split=0.1,
    rescale=1.0 / 255
)



def load_train(path):
    features_train = datagen.flow_from_dataframe(
        pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=TO_IMG_SIZE,
        batch_size=16,
        class_mode='raw',
        color_mode=COLOR_MODE,
        subset='training',
        seed=SEED
    )

    return features_train


def load_test(path):

    features_test = datagen.flow_from_dataframe(
        pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=TO_IMG_SIZE,
        batch_size=16,
        class_mode='raw',
        color_mode=COLOR_MODE,
        subset='validation',
        seed=SEED
    )

    return features_test


def create_model(input_shape):
    backbone = ResNet50(input_shape=(IMG_SIDE, IMG_SIDE, 3),
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)

    # замораживаем ResNet50 без верхушки if trainable is False
    backbone.trainable = True

    model = keras.models.Sequential()
    model.add(backbone)

    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics=['mae']
    )

    print(model.summary())
    return model


def train_model(model,
                features_train,
                features_test,
                batch_size=None,
                epochs=20,
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
              )

    return model
