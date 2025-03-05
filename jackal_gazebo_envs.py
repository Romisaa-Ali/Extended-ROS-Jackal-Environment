import gym
import time
import numpy as np
import os
from os.path import join
import subprocess
from gym.spaces import Box

try:  # make sure to create a fake environment without ros installed
    import rospy
    import rospkg
except ModuleNotFoundError:
    pass

from envs.gazebo_simulation import GazeboSimulation


class JackalGazebo(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.gui = kwargs.get('gui', False)
        self.verbose = kwargs.get('verbose', True)
        self.init_sim = kwargs.get('init_sim', True)
        self.world_name = kwargs.get('world_name', "jackal_world.world")
        self.init_position = kwargs.get('init_position', [0, 0, 0])
        self.time_step = kwargs.get('time_step', 1)
        self.max_step = kwargs.get('max_step', 100)
        self.slack_reward = kwargs.get('slack_reward', -1)
        self.failure_reward = kwargs.get('failure_reward', -50)
        self.success_reward = kwargs.get('success_reward', 0)
        self.collision_reward = kwargs.get('collision_reward', 0)
        self.goal_reward = kwargs.get('goal_reward', 1)
        self.max_collision = kwargs.get('max_collision', 10000)

        if self.init_sim:
            rospy.logwarn(f">>>>>>>>>>>>>>>>> Load world: {self.world_name} <<<<<<<<<<<<<<<<<<")
            rospack = rospkg.RosPack()
            self.BASE_PATH = rospack.get_path('jackal_helper')
            world_name = join(self.BASE_PATH, "worlds", self.world_name)
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_launch.launch')

            self.gazebo_process = subprocess.Popen(
                ['roslaunch', launch_file, 'world_name:=' + world_name, 'gui:=' + ("true" if self.gui else "false"), 'verbose:=' + ("true" if self.verbose else "false")]
            )
            time.sleep(10)
            rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
            rospy.set_param('/use_sim_time', True)

            # Pass world_name to GazeboSimulation
            self.gazebo_sim = GazeboSimulation(world_name=self.world_name, init_position=self.init_position)

        self.action_space = None
        self.observation_space = None
        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = self.current_time = None
        self.printed = False

    def reset(self):
        self.init_position, self.goal_position = self.gazebo_sim.generate_valid_start_goal_positions(min_distance=8)
        self.world_frame_goal = (self.goal_position[0], self.goal_position[1])
        self.gazebo_sim.set_start_goal_positions(self.init_position, self.goal_position)
        self.gazebo_sim.reset()
        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = rospy.get_time()
        self.current_time = self.start_time
        self.gazebo_sim.pause()
        pos, psi = self._get_pos_psi()
        obs = self._get_observation(pos, psi, np.zeros(self.action_space.shape))
        self.last_goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        return obs

    def step(self, action):
        """Take an action and step the environment"""
        self._take_action(action)
        self.step_count += 1
        pos, psi = self._get_pos_psi()

        self.gazebo_sim.unpause()
        # Compute observation
        obs = self._get_observation(pos, psi, action)
        
        # Compute termination
        flip = pos.z > 0.1  # Robot flip
        
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        distance_to_goal = np.linalg.norm(goal_pos)

        # Adjust success condition to ensure the distance to the goal is less than 0.4 meters
        success = distance_to_goal < 2

        timeout = self.step_count >= self.max_step

        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        self.collision_count += int(collided)

        done = flip or success or timeout or self.collision_count >= self.max_collision

        # Compute reward
        rew = self.slack_reward
        if done and not success:
            rew += self.failure_reward
        if success:
            rew += self.success_reward
        if collided:
            rew += self.collision_reward

        rew += (np.linalg.norm(self.last_goal_pos) - distance_to_goal) * self.goal_reward
        self.last_goal_pos = goal_pos

        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=goal_pos,
            time=self.current_time - self.start_time,
            success=success,
            world=self.world_name
        )
        if done:
            bn, nn = self.gazebo_sim.get_bad_vel_num()

        self.gazebo_sim.pause()

        return obs, rew, done, info

    def _take_action(self, action):
        current_time = rospy.get_time()
        while current_time - self.current_time < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()
        self.current_time = current_time
        
    def seed(self, seed):
        np.random.seed(seed) 

    def _get_observation(self, pos, psi):
        raise NotImplementedError()
    
    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position
        
        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        assert -np.pi <= psi <= np.pi, psi
        
        return pos, psi

    def close(self):
        # These will make sure all the ros processes being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")




class JackalGazeboLaser(JackalGazebo):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip
        
        obs_dim = 720 + 2 + self.action_dim  # 720 dim laser scan + goal position + action taken in this time step 
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_observation(self, pos, psi, action):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip * 2 # scale to (-1, 1)
        
        goal_pos = self.transform_goal(self.world_frame_goal, pos, psi) / 5.0 - 1  # roughly (-1, 1) range
        
        bias = (self.action_space.high + self.action_space.low) / 2.
        scale = (self.action_space.high - self.action_space.low) / 2.
        action = (action - bias) / scale
        
        obs = [laser_scan, goal_pos, action]
        
        obs = np.concatenate(obs)

        return obs
    
    def transform_goal(self, goal_pos, pos, psi):
        """ transform goal in the robot frame
        params:
            pos_1
        """
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        return lg         
