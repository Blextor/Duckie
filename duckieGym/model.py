from keras import regularizers
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout

from logger_callback import LoggerCallback
from callbacks import *
from data_reader import *


def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4)))  # TODO kernel size
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.15))

    model.add(Dense(12000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="linear"))

    return model


def train_model(model, X_train, Y_train, X_valid, Y_valid):
    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=["mse"])

    print(model.summary())

    model.fit(X_train, Y_train, batch_size=32, epochs=10000, validation_data=(X_valid, Y_valid),
              callbacks=[early_stopping, reduce_lr, checkpoint, change_lr, LoggerCallback("stat.csv")], verbose=1,
              shuffle=True)


X, Y = read_data()

(X_scaled, Y_scaled), velocity_steering_scaler = scale(X, Y)

(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_data(X_scaled, Y_scaled)

model = create_model(X_train[0].shape)

train_model(model, X_train, Y_train, X_valid, Y_valid)

print("eval score: ", model.evaluate(X_test, Y_test, batch_size=32))

y_test_pred = model.predict(X_test)

for idx, pred in enumerate(y_test_pred):  # TODO maybe pyplot history after training
    print(pred, Y_test[idx])
