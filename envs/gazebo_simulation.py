import numpy as np


try:
    import rospy
    from std_srvs.srv import Empty
    from gazebo_msgs.msg import ModelState, ModelStates
    from gazebo_msgs.srv import SpawnModel, SetModelState, GetWorldProperties, DeleteModel, GetModelState
    from geometry_msgs.msg import Quaternion, Twist, Pose  # Import Pose here
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    pass

import numpy as np
import random
import math

from actor import get_world_name

from plot_points3 import PositionChecker


def zone_to_coordinates(zone):
    col = (zone - 1) % 16
    row = 15 - (zone - 1) // 16
    x = col - 8
    y = row - 8
    return x, y


def create_model_state(x, y, z, angle):
    model_state = ModelState()
    model_state.model_name = 'jackal'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    model_state.pose.orientation = Quaternion(0, 0, np.sin(angle / 2.), np.cos(angle / 2.))
    model_state.reference_frame = "world"
    return model_state


def is_far_enough(new_pose, invalid_zones, invalid_zone_margin=1.7):
    for zone in invalid_zones:
        zone_x, zone_y = zone_to_coordinates(zone)
        dist = math.sqrt((new_pose.position.x - zone_x) ** 2 +
                         (new_pose.position.y - zone_y) ** 2)
        if dist < invalid_zone_margin:
            return False
    return True


class GazeboSimulation:
    def __init__(self, world_name, init_position=[0, 0, 0]):
        self.cleaned_world_name = world_name.replace("BARN/", "").replace(".world", "")
        
        # Service proxies
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Initial model state
        self._init_model_state = create_model_state(init_position[0], init_position[1], 0, init_position[2])
        
        # Collision tracking
        self.collision_count = 0
        self._collision_sub = rospy.Subscriber("/collision", Bool, self.collision_monitor)
        
        # Velocity monitoring
        self.bad_vel_count = 0
        self.vel_count = 0
        self._vel_sub = rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.vel_monitor)
        
        # Start and goal positions
        self._start_position = None
        self._goal_position = None
        
        # Position checker
        self.position_checker = PositionChecker(world=self.cleaned_world_name)



    def generate_valid_start_goal_positions(self, min_distance=8, max_distance=11, size=16):
        while True:
            start_position = self.position_checker.generate_random_position(size)
            goal_position = self.position_checker.generate_random_position(size)
        
            distance = self.position_checker.distance(start_position[:2], goal_position[:2])

            start_pose = Pose()
            start_pose.position.x = start_position[0]
            start_pose.position.y = start_position[1]

            goal_pose = Pose()
            goal_pose.position.x = goal_position[0]
            goal_pose.position.y = goal_position[1]

            # Check if start and goal positions are far enough from invalid zones
            if is_far_enough(start_pose, self.position_checker.invalid_zones) and \
               is_far_enough(goal_pose, self.position_checker.invalid_zones) and \
               min_distance <= distance <= max_distance:
                return start_position, goal_position




    def set_start_goal_positions(self, start_position, goal_position):
        self._init_model_state = create_model_state(start_position[0], start_position[1], 0, start_position[2])
        self._start_position = start_position
        self._goal_position = goal_position
        self.reset()  # Immediately reset the model state to start position

    def vel_monitor(self, msg):
        """
        Count the number of velocity command and velocity command
        that is smaller than 0.2 m/s (hard coded here, count as self.bad_vel)
        """
        vx = msg.linear.x
        if vx <= 0:
            self.bad_vel_count += 1
        self.vel_count += 1
        
    def get_bad_vel_num(self):
        """
        return the number of bad velocity and reset the count
        """
        bad_vel = self.bad_vel_count
        vel = self.vel_count
        self.bad_vel_count = 0
        self.vel_count = 0
        return bad_vel, vel
        
    def collision_monitor(self, msg):
        if msg.data:
            self.collision_count += 1
    
    def get_hard_collision(self):
        # hard collision count since last call
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided

    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")

    def unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

    def reset(self):
        """
        /gazebo/reset_world or /gazebo/reset_simulation will
        destroy the world setting, here we used set model state
        to put the model back to the origin
        """
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._reset(self._init_model_state)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/set_model_state service call failed")

    def get_laser_scan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass
        return data

    def get_model_state(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            return self._model_state_getter('jackal', 'world')
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed")

    def reset_init_model_state(self, init_position = [0, 0, 0]):
        """Overwrite the initial model state

        Args:
            init_position (list, optional): initial model state in x, y, z. Defaults to [0, 0, 0].
        """
        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])
