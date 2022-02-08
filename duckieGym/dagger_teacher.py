import math
import numpy as np
from gym_duckietown.simulator import AGENT_SAFETY_RAD

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15


class DaggerTeacher:
    """
    A Pure Pusuit controller class to act as an expert to the model
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images
    loss(*args)
        takes images and target action to compute the loss function used in optimization
    predict(observation)
        takes an observation image and predicts using env information the action
    """

    def __init__(
            self, env, ref_velocity=REF_VELOCITY, following_distance=FOLLOWING_DISTANCE, max_iterations=1000
    ):
        """
        Parameters
        ----------
        ref_velocity : float
            duckiebot maximum velocity (default 0.7)
        following_distance : float
            distance used to follow the trajectory in pure pursuit (default 0.24)
        """
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity

    def predict(self, deleteThis, observation):
        """
        Parameters
        ----------
        observation : image
            image of current observation from simulator
        Returns
        -------
        action: list
            action having velocity and omega of current observation
        """
        closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(
            self.env.cur_pos, self.env.cur_angle
        )
        if closest_point is None or closest_tangent is None:
            self.env.reset()
            closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(
                self.env.cur_pos, self.env.cur_angle
            )

        current_world_objects = self.env.objects
        # to slow down if there's a duckiebot in front of you
        # this is used to avoid hitting another moving duckiebot in the map
        # in case of training LFV baseline
        velocity_slow_down = 1
        for obj in current_world_objects:
            if not obj.static and obj.kind == "duckiebot":
                if True:
                    collision_penalty = abs(
                        obj.proximity(self.env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN)
                    )
                    if collision_penalty > 0:
                        # this means we are approaching and we need to slow down
                        velocity_slow_down = collision_penalty
                        break

        lookup_distance = self.following_distance
        # projected_angle used to detect corners and to reduce the velocity accordingly
        projected_angle, _, _ = self._get_projected_angle_difference(0.3)
        velocity_scale = 1

        current_tile_pos = self.env.get_grid_coords(self.env.cur_pos)
        current_tile = self.env._get_tile(*current_tile_pos)
        if "curve" in current_tile["kind"] or abs(projected_angle) < 0.92:
            # slowing down by a scale of 0.5
            velocity_scale = 0.5
        _, closest_point, curve_point = self._get_projected_angle_difference(lookup_distance)

        if closest_point is None:  # if cannot find a curve point in max iterations
            return [0, 0]

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        right_vec = np.array([math.sin(self.env.cur_angle), 0, math.cos(self.env.cur_angle)])
        dot = np.dot(right_vec, point_vec)
        omega = -1 * dot
        # range of dot is just -pi/2 and pi/2 and will be multiplied later by a gain adjustable if we are testing on a hardware or not
        velocity = self.ref_velocity * velocity_scale
        if velocity_slow_down < 0.2:
            velocity = 0
            omega = 0

        action = [velocity, omega]
        print("Teacher predict: ",velocity,omega)
        return action

    def _get_projected_angle_difference(self, lookup_distance):
        # Find the projection along the path
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        curve_angle = None

        while iterations < 10:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_angle = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_angle is not None and curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_angle is None:  # if cannot find a curve point in max iterations
            return None, None, None

        else:
            return np.dot(curve_angle, closest_tangent), closest_point, curve_point


"""
import numpy as np
import math

from gym_duckietown.envs import DuckietownEnv


class DaggerTeacher():
    def predict(self,environment, obs):
        ""
        Implement pure-pursuit & PID using ground truth
        Returns [velocity, steering]
        ""

        env = environment

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
        #steering *= 1.4

        if steering < 0.1 and steering > -0.1:
            velocity = 0.6
        else:
            velocity = 0.4

            # TODO The numbers above could be altered, works great
        return [velocity, steering]
"""
