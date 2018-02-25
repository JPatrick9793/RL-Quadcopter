"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_distance = 25.0
        self.max_duration = 7.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        
        self.target_vector = np.array([0.0, 0.0, 10.0]) # constrain the position to be directly above starting point
        self.target_pose = np.array([0.0, 0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])

        self.w_vec = 4.0
        self.w_pose = 0.0
        self.w_vel = 1.0

        self.last_timestamp = None
        self.last_position = None


    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)
        
        # set state variale
        state = np.concatenate([position, orientation, velocity]) # add in velocity
        
        # Reset instance variables
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False

        # reward is negative square abs distance
        euc_distance = np.linalg.norm(self.target_vector-state[0:3])
        euc_orientation = np.linalg.norm(self.target_pose-state[3:7])
        euc_velocity = np.linalg.norm(self.target_velocity-state[7:10])

        # print ("euc_distance:\t{0}".format(euc_distance))
        reward = -(
            self.w_vec  * euc_distance +
            self.w_pose * euc_orientation +
            self.w_vel  * euc_velocity)

        # if quad copter is close enough, increase reward
        if euc_distance < 1.0:
            reward += 10.0

        # if distance is too great, end episode and add penalty
        if euc_distance >= self.max_distance:
            reward -= 1000.0
            done = True 

        # if timestamp exceeds time limit, end episode
        if timestamp > self.max_duration:
            done = True
        # print ("State sent:\n{0}".format(state))
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)
        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            # print ("Action is not None!")
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            print ("Action is None?")
            return Wrench(), done
