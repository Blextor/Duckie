import os
import time

import keras


class LoggerCallback(keras.callbacks.Callback):

    def __init__(self, file_name):
        self.file_name = file_name
        self.columns_done = False
        self.id = int(time.time())

    def on_epoch_end(self, epoch, logs=None):
        with open(os.path.join(os.getcwd(), 'stats', str(self.id) + str(self.file_name)), "a") as f:
            keys = list(logs.keys())
            if not self.columns_done:
                for k in keys:
                    f.write(str(k) + " , ")
                f.write("\n")
                self.columns_done = True

            for k in keys:
                f.write(str(logs[k]) + " , ")
            f.write("\n")
