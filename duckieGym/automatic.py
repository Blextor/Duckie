#!/usr/bin/env python3

"""
This is based on Frank's script to log runs using ground truth 
"""

import argparse
import cv2
import gym
import math
import numpy as np
import pyglet
import sys
import time

from log_util import Logger
from log_schema import Episode, Step

from typing import List
from gym_duckietown.envs import DuckietownEnv

REWARD_INVALID_POSE = -1000


class DataGenerator:
    def __init__(self, env, max_episodes, max_steps, log_file=None, downscale=False):
        if not log_file:
            log_file = f"dataset.log"
        self.env = env
        self.counter = 0
        self.env.reset()
        self.logger = Logger(self.env, log_file=log_file)
        self.episode = 1
        self.max_episodes = max_episodes
        self.downscale = downscale

        # ! Enter main event loop
        print("Starting data generation")

        pyglet.clock.schedule_interval(
            self.update, 1.0 / self.env.unwrapped.frame_rate, self.env
        )

        pyglet.app.run()

        print("App exited, closing file descriptors")
        self.logger.close()
        self.env.close()

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def pure_pursuite(self, env) -> List[float]:
        """
        Implement pure-pursuit & PID using ground truth
        Returns [velocity, steering]
        """

        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = env.closest_curve_point(
            env.cur_pos, env.cur_angle
        )

        iterations = 0

        lookup_distance = 0.25  # CHANGED from 0.5
        max_iterations = 1000
        gain = 4.0  # 2.0
        velocity = 0.4  # CHANGED from 0.35
        curve_point = None

        while iterations < max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = env.closest_curve_point(follow_point, env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        right_vec = [math.sin(env.cur_angle), 0, math.cos(env.cur_angle)]

        dot = np.dot(right_vec, point_vec)
        steering = gain * -dot

        # ADDED THE FOLLOWING LINES TILL RETURN
        steering *= 1.4

        if steering < 0.1 and steering > -0.1:
            velocity = 0.6
        else:
            velocity = 0.4

            # TODO The numbers above could be altered, works great
        return [velocity, steering]

    def update(self, dt, env):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        action = self.pure_pursuite(env)
        # action = self.expert_policy.predict("not used param TODO")

        # ! GO! and get next
        # * Observation is 640x480 pixels
        obs, reward, done, info = env.step(action)

        if reward == REWARD_INVALID_POSE:
            print("Out of bound")
        else:
            output_img = obs
            if self.downscale:
                # Resize to (150x200)
                # ! resize to Nvidia standard:
                obs_distorted_DS = self.image_resize(obs, width=200)

                # ! ADD IMAGE-PREPROCESSING HERE!!!!!
                height, width = obs_distorted_DS.shape[:2]
                # print('Distorted return image Height: ', height,' Width: ',width)
                cropped = obs_distorted_DS[0:150, 0:200]

                # NOTICE: OpenCV changes the order of the channels !!!
                output_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                # print(f"Recorded shape: {obs.shape}")
                # print(f"Saved image shape: {cropped.shape}")

            step = Step(output_img, reward, action, done)

            self.logger.log(step, info)
            self.counter=0
            # rawlog.log(obs, action, reward, done, info)
            # last_reward = reward

        if done:
            self.logger.on_episode_done()
            print(f"episode {self.episode}/{self.max_episodes}")
            self.episode += 1
            env.reset()
            if self.logger.episode_count >= args.nb_episodes:
                print("Training completed !")
                sys.exit()
            time.sleep(1)
            return


if __name__ == "__main__":
    # ! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="small_loop")
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
        "--steps", default=200, help="number of steps to record in one batch", type=int
    )
    parser.add_argument("--nb-episodes", default=1, type=int)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--downscale", action="store_true")

    args = parser.parse_args()

    # ! Start Env
    if args.env_name is None:
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
    else:
        env = gym.make(env=args.env_name)

    node = DataGenerator(env, max_episodes=args.nb_episodes, max_steps=args.steps, log_file=args.logfile,
                         downscale=args.downscale)
