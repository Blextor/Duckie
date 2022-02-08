#!/usr/bin/env python

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import sys
import os
import argparse
import math
import json
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from PIL import Image

from gym_duckietown.envs import DuckietownEnv
from keras.models import load_model
from tensorflow import keras
from matplotlib import image

from dagger_learner import DaggerLearner
from detector import preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='zigzag_dists')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument("--model_path", default=None)
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        distortion= args.distortion,
        domain_rand = args.domain_rand,
        max_steps = np.inf
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()
model = load_model(args.model_path)
learner=DaggerLearner(model)
first = False

positions = []
actions = []
demos = []
recording = False

def write_to_file(demos):
    num_steps = 0
    for demo in demos:
        num_steps += len(demo['actions'])
    print('num demos:', len(demos))
    print('num steps:', num_steps)

    # Store the trajectories in a JSON file
    with open('experiments/demos_{}.json'.format(args.map_name), 'w') as outfile:
        json.dump({ 'demos': demos }, outfile)

def process_recording():
    global positions, actions, demos

    if len(positions) == 0:
        # Nothing to delete
        if len(demos) == 0:
            return

        # Remove the last recorded demo
        demos.pop()
        write_to_file(demos)
        return

    p = list(map(lambda p: [ p[0].tolist(), p[1] ], positions))
    a = list(map(lambda a: a.tolist(), actions))

    demo = {
        'positions': p,
        'actions': a
    }

    demos.append(demo)

    # Write all demos to this moment
    write_to_file(demos)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    #cec = env.getObs()
    cec, reward, done, info = env.step([0.0,0.0])
    action=learner.predict(env,cec)
    action = np.array(action)
    action[0] *= 1.0
    action[1]*=1.0

    if recording:
        positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))
        actions.append(action)

    cec, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()

        if recording:
            process_recording()
            positions = []
            actions = []
            print('Saved Recoding')

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Registers joysticks and recording controls
#joysticks = pyglet.input.get_joysticks()
#assert joysticks, 'No joystick device is connected'
#joystick = joysticks[0]
#joystick.open()
#joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()

env.close()
