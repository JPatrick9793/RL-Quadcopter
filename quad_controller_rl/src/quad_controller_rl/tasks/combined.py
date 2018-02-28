# TASK FOR COMBINED TAKEOFF, HOVER, AND LANDING
import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force,       -5.0, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        # Task-specific parameters
        self.max_distance = 20.0
        self.max_duration = 10.0
        self.min_duration = 2.0

        self.last_timestamp = None
        self.last_position = None
        
        # 0 = takeoff / hover
        # 1 = landing
        self.which_task = 0                # number to indicate which task the agent 
        self.switch_task_after_eps = 5     # after 5 epsidos, switch tasks

    def get_which_task(self):
        return self.which_task

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        # if takeoff task, start from ground
        if self.which_task == 0:
            return Pose(position=Point(0.0, 0.0, np.random.normal(2.0, 0.25)),
                    orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                ), Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))
        # if landing task, start from in air
        else:
            return Pose(
                    position=Point(0.0, 0.0, np.random.normal(10.0, 0.25)),
                    orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                ), Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(abs(timestamp - self.last_timestamp),
                                                             1e-05)
        
        # set state variale
        state = np.concatenate([position, orientation, velocity])       # add in velocity
        
        self.last_timestamp = timestamp                                 # Reset instance variables
        self.last_position = position
        done = False                                                    # initilize done
        
        ###############################
        # CALCULATE INDIVIDUAL ERRORS #
        ###############################
        
        # if task = takeoff
        if self.which_task == 0:         
            x = np.linalg.norm(np.array([0.0, 0.0, 10.0])      - state[0:3])     # distance between copter and origin
            o = np.linalg.norm(np.array([0.0, 0.0,  0.0, 0.0]) - state[3:7])     # NOT NEEDED BECAUSE SPACE CONSTRAINED
            v = np.linalg.norm(np.array([0.0, 0.0,  0.0])      - state[7:10])    # velocity magnitude
            
            v_error = ((x*3)**2 + v**2)
            x_error = abs(x**2)                        # error is abs distance from 0.0
            # v_error = abs(v)/(abs(x) + 0.1)          # prevent divide by 0
            
            reward = -v_error                          # x and v errors
            reward = (1/1000)*reward                   # scale down so no exploding gradients
            
            if state[2] == 10.0:                       # if the quadcopter hits target
                reward += 50                           # give give reward
            if x > self.max_distance:                  # if max distance is exceeded
                done = True                            # end current episode
                reward -= 100                          # give penalty
            
        # if task = landing
        if self.which_task == 1:         
            x = np.linalg.norm(np.array([0.0, 0.0,  2.0])      - state[0:3])     # distance between copter and origin
            o = np.linalg.norm(np.array([0.0, 0.0,  0.0, 0.0]) - state[3:7])     # NOT NEEDED BECAUSE SPACE CONSTRAINED
            v = np.linalg.norm(np.array([0.0, 0.0,  0.0])      - state[7:10])    # velocity magnitude
            
            v_error = (x**2 + (3*v)**2)
            x_error = abs(x**2)                        # error is abs distance from 0.0
            # v_error = abs(v)/(abs(x) + 0.1)          # prevent divide by 0
            
            reward = -(0.25)*x_error - (0.75)*v_error  # x and v errors
            reward = (1/1000)*reward                   # scale down so no exploding gradients
            
            if state[2] == 2.0:                        # if the quadcopter hits target
                reward += 50                           # give give reward
            if x > self.max_distance:                  # if max distance is exceeded
                done = True                            # end current episode
                reward -= 50                           # give penalty
        
        # CHECK IF TIME LIMIT EXCEEDED
        if timestamp > self.max_duration:     # if time limit is exceeded
            done = True                       # end current episode
        
        # RESET EPISODE COUNT WHEN SWITCHING TASKS
        if self.agent.get_epCount() == self.switch_task_after_eps:
            self.which_task = 1 if self.which_task == 0 else 0
            self.agent.reset_epCount()
        
        # GET ACTION
        action = self.agent.step(state, reward, done)

        ###################################################################
        # Convert to proper force command (a Wrench object) and return it #
        ###################################################################
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
