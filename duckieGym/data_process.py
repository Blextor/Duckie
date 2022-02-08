"""
1635618838929 0.4 -4.591255574309327
1635618839044 0.4 -4.591255574309327
1635618839145 0.4 -4.591255574309327
1635618839244 0.4 -4.591255574309327
1635618839349 0.4 -4.591255574309327
"""
from cv2 import imread
import os
import numpy as np

val_ratio = 0.1
train_ratio = 0.7
test_ratio = 0.2


def process():
    # X = images
    # Y = [steer],[velocity]
    X = []
    y = []
    with open(os.path.join(os.getcwd(), "my_app.txt")) as file:
        lines = file.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) != 3:
                continue
            time, steer, velocity = map(float, data)
            X.append((steer, velocity))
            img_path = os.path.join(os.getcwd(), "myapp", str(int(time)) + ".png")
            img = imread(img_path)
            y.append(img)


    X = np.asarray(X)
    y = np.asarray(y)
    shuffled_indices = np.random.permutation(len(X))
    # TRAIN-VAL-TEST
    train_end = int(len(shuffled_indices) * train_ratio)
    val_end = int(len(shuffled_indices) * (train_ratio + val_ratio))
    X_train = X[shuffled_indices[:train_end]]
    y_train = y[shuffled_indices[:train_end]]
    X_val = X[shuffled_indices[train_end: val_end]]
    y_val = y[shuffled_indices[train_end: val_end]]
    X_test = X[shuffled_indices[val_end:]]
    y_test = y[shuffled_indices[val_end:]]

    print("indices")
    print("test",X_test)
    print("train",X_train)
    print("val",X_val)

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = process()
    print(os.getcwd())
