from keras_tuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
# from keras.models import load_model
from tensorflow import keras

from keras import regularizers
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout

from callbacks import *
from data_reader import *
from duckieGym.logger_callback import LoggerCallback

(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = create_x_y()

input_shape2 = X_train[0].shape
input_xtest = X_test[0].shape
input_shape3 = X_valid[0].shape
print(input_shape2, input_xtest, input_shape3)


def build_model(hp):
    model = Sequential()

    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     input_shape=input_shape2,
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.2)))

    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.2)))

    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.2)))

    model.add(Dense(hp.Int("dense_num", min_value=2000, max_value=12000, step=2000),
                    activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    ))
    model.add(Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.2)))
    model.add(Dense(2, activation="linear"))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])),
                  loss='mse',
                  metrics=['mse'],
                  )

    return model


def hyperopti():
    tuner = Hyperband(
        build_model,
        objective='val_loss',
        factor=3,
        max_epochs=10,
        directory='./duckieGym/hyperopti10')

    tuner.search_space_summary()

    early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min')

    tuner.search(X_train, Y_train,
                 epochs=100, validation_split=0.1,
                 callbacks=[early_stopping]
                 )

    best_model = tuner.get_best_models(num_models=1)[0]
    # best_model.summary()

    params_best = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print(params_best.get_config()['values'])

    # tuner.results_summary()

    model_best = tuner.hypermodel.build(params_best)
    history = model_best.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid),
                             callbacks=[early_stopping, reduce_lr, checkpoint, change_lr, LoggerCallback("stat.csv")],
                             verbose=1,
                             shuffle=True)

    saved = keras.models.load_model('duckie.hdf5')

    eval_result = saved.evaluate(X_test, Y_test)
    print("[test loss, test accuracy]:", eval_result)

    # model_best.save("/tmp/model")

    # print(model_best.predict(X_train[0]))

    return model_best


result = hyperopti()
model = keras.models.load_model('duckie.hdf5')
print(model.summary())
