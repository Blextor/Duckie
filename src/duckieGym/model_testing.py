#!/usr/bin/env python3

"""
This is based on Frank's script to log runs using ground truth 
"""

import argparse
import tensorflow as tf
import os
from PIL import Image
import cv2
import gym
import math
import numpy as np
import pyglet
import sys
import time
from pathlib import Path

from keras.models import load_model

from duckieGym.data_reader import read_data, scale
from duckieGym.detector import preprocess_image
from log_util import Logger
from log_schema import Episode, Step

from typing import List
from gym_duckietown.envs import DuckietownEnv

REWARD_INVALID_POSE = -1000

"""
Class used to test the model in the environment. Once __init__ is called it starts a simulation 
and saves the observed images. Also is a wrapper class for the model used for making predictions
"""


class ModelTestEnvironment:
    def __init__(self, env, max_episodes, max_steps, model, log_file=None, downscale=False):
        if not log_file:
            log_file = f"dataset.log"
        self.env = env
        self.model = model
        self.counter = 0
        self.env.reset()
        self.logger = Logger(self.env, log_file=log_file)
        self.episode = 1
        self.max_episodes = max_episodes
        self.downscale = downscale
        self.max_steps = max_steps
        self.curr_obs = None
        self.log_dir = os.path.join(os.getcwd(), "test_log")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.step = 0
        # ! Enter main event loop
        print("Starting data generation")

        pyglet.clock.schedule_interval(
            self.update, 1.0 / self.env.unwrapped.frame_rate, self.env
        )

        pyglet.app.run()

        print("App exited, closing file descriptors")
        self.logger.close()
        self.env.close()

    def get_action(self, curr_obs):
        x = preprocess_image(curr_obs)
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = x / 255.0

        x = np.reshape(x, (1, 48, 85, 3))
        y = self.model.predict(x)

        return (y[0][0] * 1.6, y[0][1] * 10)

    def save_obs(self, obs):

        img = Image.fromarray(obs)

        path = os.path.join(self.log_dir, str(self.step) + ".png")
        img.save(path)

    def update(self, dt, env):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        if self.curr_obs is None:
            obs, reward, done, info = env.step([0.0, 0.0])
            self.curr_obs = obs
            self.save_obs(self.curr_obs)

        else:
            action = self.get_action(self.curr_obs)
            self.save_obs(self.curr_obs)
            obs, reward, done, info = env.step(action)
            self.curr_obs = obs

        self.step += 1
        print("currently at frame", self.step)
        if self.step >= self.max_steps:
            exit(1)

"""
def test_model_in_environment():
    # ! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="zigzag_dists")
    parser.add_argument(
        "--draw-curve", default=False, help="draw the lane following curve"
    )
    parser.add_argument(
        "--draw-bbox", default=False, help="draw collision detection bounding boxes"
    )
    parser.add_argument(
        "--domain-rand", default=False, help="enable domain randomization"
    )
    parser.add_argument("--distortion", default=True)

    parser.add_argument(
        "--raw-log", default=False, help="enables recording high resolution raw log"
    )
    parser.add_argument(
        "--steps", default=1000, help="number of steps to record in one batch", type=int
    )
    parser.add_argument("--nb-episodes", default=1, type=int)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--downscale", action="store_true")

    args = parser.parse_args()

    env = DuckietownEnv(
        map_name=args.map_name,
        max_steps=args.steps,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        distortion=args.distortion,
        accept_start_angle_deg=4,
        full_transparency=True,
    )

    node = ModelTestEnvironment(env, max_episodes=args.nb_episodes, model=loaded_model, max_steps=args.steps,
                                log_file=args.logfile,
                                downscale=args.downscale)

"""
if __name__ == "__main__":

    loaded_model = load_model("best_model")

    # uncomment for testing in the environment
    # test_model_in_environment(loaded_model)

    maps_to_test = ("canyon", "small_loop", "zigzag_dists")

    for map_name in maps_to_test:
        x_path = os.path.join(os.getcwd(), "test_set", map_name)
        labels = os.path.join(os.getcwd(), "test_set",map_name + ".txt")

        X, Y = read_data(images_dir_name=x_path, label_file=labels)
        (X_test, Y_test), velocity_steering_scaler = scale(X, Y)

        y_pred = loaded_model.predict(X_test)
        y_pred[:,0]*=1.6
        y_pred[:,1]*=10.0

        mse = sum((y_pred-Y_test)**2)/len(y_pred)
        print("MAP",map_name)
        print("    eval score for map", map_name, ":", loaded_model.evaluate(X_test, Y_test, batch_size=32))
        print("    MSE for steering using adjusted results",mse[0])
        print("    MSE for velocity using adjusted results",mse[1])