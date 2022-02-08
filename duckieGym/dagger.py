import argparse
import os
import time

import numpy as np
from gym_duckietown.envs import DuckietownEnv
from keras.models import load_model, Sequential
from tensorflow import keras

from dagger_learner import DaggerLearner
from dagger_teacher import DaggerTeacher
from IIL import InteractiveImitationLearning
from dagger_sandbox import MyInteractiveImitationLearning
from detector import preprocess_image
from data_reader import *
from tensorflow.keras.callbacks import EarlyStopping
from callbacks import *
from modelfit import *

class DAgger(MyInteractiveImitationLearning):
    """
    DAgger algorithm to mix policies between learner and expert
    Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." Proceedings of the fourteenth international conference on artificial intelligence and statistics. 2011.
    ...
    Methods
    -------
    _mix
        used to return a policy teacher / expert based on random choice and safety checks
    """

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5, test=False):
        MyInteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, test)
        # expert decay
        self.p = alpha
        self.alpha = self.p
        self.counter=0

        self.teacherDecisions = 0
        self.learnerDecisions = 0
        self.learnerIdxs = []  # frame indexes during which the learner was behind the wheel

        # thresholds used to give control back to learner once the teacher converges
        self.convergence_distance = 0.05
        self.convergence_angle = np.pi / 18

        self.learner_streak = 0
        self.teacher_streak = 0

        # threshold on angle and distance from the lane when using the model to avoid going off track and env reset within an episode
        self.angle_limit = np.pi / 8
        self.distance_limit = 0.12

    def _mix(self):
        control_policy = self.learner
        #return control_policy
        # control_policy = self.learner  #swapped from: np.random.choice(a=[self.teacher, self.learner], p=[self.alpha, 1.0 - self.alpha])

        if self.learner_streak > 50:
            self.learner_streak = 0
            return self.teacher

        if self._found_obstacle:
            self.learner_streak = 0
            return self.teacher
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except:
            return control_policy
        if self.active_policy:
            # keep using teacher until duckiebot converges back on track
            if not (abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad) < self.convergence_angle):
                self.learner_streak = 0
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give
            # control back to the expert
            if abs(lp.dist) > self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                self.learner_streak = 0
                return self.teacher

        self.learnerIdxs.append(self.learnerDecisions + self.teacherDecisions)
        self.learner_streak += 1
        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)

    @property
    def observations(self):
        return self._observations


if __name__ == "__main__":
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
    parser.add_argument("--model_path", default=None)

    args = parser.parse_args()

    # ! Start Env


    print("alma")
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = create_x_y()

    print("balma")
    model = load_model(args.model_path)
    #model = Sequential()
    #model.load_weights(args.model_path)
    eval_result = model.evaluate(X_test, Y_test)
    print("[test loss, test accuracy]:", eval_result)
    print("calma")

    dagger_run_dir = os.path.join("daggerObservations",str(round(time.time() * 1000)) )
    os.mkdir(dagger_run_dir)
    print("dalma")
    for i in range(40):
        env = DuckietownEnv(
            map_name=str(i+10),
            max_steps=args.steps,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            distortion=args.distortion,
            accept_start_angle_deg=4,
            full_transparency=True,
        )
        iil = DAgger(env=env, teacher=DaggerTeacher(env), learner=DaggerLearner(model), horizon=100 * 10, episodes=1)
        print("ilma")
        #n_dagger_runs = 10
        dagger_run_dir2 = os.path.join(dagger_run_dir, str(i))
        os.mkdir(dagger_run_dir2)
        #for run in range(n_dagger_runs):
         #   print("Running Dagger... Run:", run)

        # run dagger
        iil.train()

        # get and save images
        observation = iil.get_observations()

        # get labels from expert
        labels = iil.get_expert_actions()
        print("\tsaving {number} images...".format(number=len(labels)))


        image_id = []
        counter = 0
        for id, obs in enumerate(observation):
            if (counter<1):
                img = preprocess_image(obs)
                path = os.path.join(os.getcwd(), dagger_run_dir2)
                time_now = str(round(time.time() * 1000))
                image_id.append(time_now)
                img.save(os.path.join(path, time_now + ".png"))
            counter+=1
            if (counter>4):
                counter=0

        filepath = os.path.join(os.getcwd(), dagger_run_dir2)
        counter = 0
        with open(os.path.join(os.getcwd(), "labels.txt"), "a") as f:
            for label in labels:
                if (counter<1):
                    f.write(image_id[counter]+" "+str(label[0]) + " " + str(label[1]))
                    f.write("\n")
                counter+=1
                if (counter>4):
                    counter=0

        # train model on the new dagger data
        X, Y = read_data(dagger_run_dir2, "labels.txt")
        (X_scaled, Y_scaled), velocity_steering_scaler = scale(X, Y)

        print("\tTraining model:",i)

        modelfitsplit(model,X_scaled,Y_scaled)

        keras.models.save_model(model,args.model_path)


        eval_result = model.evaluate(X_test, Y_test)
        print("[test loss, test accuracy]:", eval_result)
