from keras_tuner.tuners import Hyperband
#from keras.models import load_model
from tensorflow import keras
from sklearn.model_selection import train_test_split


from data_reader import *


def create_x_y():
    X, Y = read_data()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

    (X_train_scaled, Y_train_scaled), velocity_steering_scaler_train = scale(X_train, y_train)
    (X_valid_scaled, Y_valid_scaled), velocity_steering_scaler_valid = scale(X_valid, y_valid)
    (X_test_scaled, Y_test_scaled), velocity_steering_scaler_test = scale(X_test, y_test)

    return (X_train_scaled, Y_train_scaled), (X_valid_scaled, Y_valid_scaled), (X_test_scaled, Y_test_scaled)


(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = create_x_y()

input_shape2 = X_train[0].shape
print(input_shape2)

def build_model(hp):
    model = Sequential()

    model.add(Conv2D(filters=hp.Int('conv_1_filter', min_value=16, max_value=96, step=16),
                     kernel_size=hp.Choice('conv_1_kernel', values=[3, 5, 7]),
                     input_shape=input_shape2
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=16, max_value=64, step=16),
                     kernel_size=hp.Choice('conv_2_kernel', values=[3, 5, 7]),
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=16),
                     kernel_size=hp.Choice('conv_3_kernel', values=[3, 5, 7]),
                     )
              )
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.9, step=0.2)))

    model.add(Dense(12000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.9, step=0.2)))
    model.add(Dense(2, activation="linear"))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[0.01,0.001, 0.0001, 0.00001])),
                  loss='mse',
                  metrics=['mse'],
                  )

    return model


def hyperopti():
    tuner = Hyperband(
        build_model,
        objective='val_loss',
        factor=3,
        max_epochs=50,
        directory='./duckieGym/hyperopti')

    tuner.search_space_summary()

    #early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min')

    #tuner.search(X_train, Y_train,
                 #epochs=100, validation_split=0.1,
                 #callbacks=[early_stopping]
                 #)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    params_best = tuner.get_best_hyperparameters(num_trials=1)[0]
    #print(params_best.get_config()['values'])

    #tuner.results_summary()

    model_best = tuner.hypermodel.build(params_best)
    #history = model_best.fit(X_train, Y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    #eval_result = model_best.evaluate(X_test, Y_test)
    #print("[test loss, test accuracy]:", eval_result)

    #model_best.save("/tmp/model")
    keras.models.save_model(model_best,"/tmp/model9")
    
    #keras.models.save_model(model_best,"/tmp/model8")
    #print(model_best.predict(X_train[0]))
    

    return model_best


result = hyperopti()
model = keras.models.load_model("/tmp/model9")
print(model.summary())
