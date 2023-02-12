import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50

optimizer = Adam(learning_rate=0.0005)
datagen = ImageDataGenerator(
    # validation_split=0.25,
    rescale=1.0 / 255
)


def load_train(path):
    features_train = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=108108108
    )

    return features_train


def create_model(input_shape):
    # ___________

    backbone = ResNet50(input_shape=(150, 150, 3),
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)

    # замораживаем ResNet50 без верхушки if trainable is False
    backbone.trainable = True

    model = keras.models.Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))

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
                epochs=10,
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
