import numpy as np

from detector import preprocess_image

import tensorflow as tf



class DaggerLearner:
    """
    Wrapper class fotr the neural network to play to role of the learner in dAgger
    """

    def __init__(self, model):
        self.model = model

    def predict(self, env, obs):
        x = preprocess_image(obs)
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = x / 255.0

        x = np.reshape(x, (1, 48, 85, 3))
        y = self.model.predict(x)

        # (velo, steer)
        print ("Learner predict: ",y[0][0],y[0][1], y)
        return (y[0][0]*2.0+0.0, y[0][1]*5.0)
