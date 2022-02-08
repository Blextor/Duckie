from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

dagger = False

def scheduler(epoch):
    if dagger:
        return 0.00002

    if epoch < 10:
        return 0.01
    if epoch < 15:
        return 0.005
    if epoch < 20:
        return 0.002
    if epoch < 25:
        return 0.001
    if epoch < 30:
        return 0.0005
    if epoch < 35:
        return 0.0002
    if epoch < 40:
        return 0.0001
    if epoch < 50:
        return 0.00005
    else:
        return 0.00002


early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)
checkpoint = ModelCheckpoint(filepath='duckie.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
change_lr = LearningRateScheduler(scheduler)
