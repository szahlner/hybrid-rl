import os
import numpy as np
import pybullet

from gym import spaces
from gym import utils

from hybridrl.environments.common.pybullet_env import PyBulletGoalEnv


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'urdf', 'panda', 'panda.urdf')
WORLD_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'urdf', 'panda', 'panda_world.urdf')
SPHERE_RED_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'urdf', 'common', 'sphere_red.urdf')

MOVABLE_JOINTS_ID = [0, 1, 2, 3, 4, 5, 6, 9, 10]
GRIPPER_JOINTS_ID = [MOVABLE_JOINTS_ID[-2], MOVABLE_JOINTS_ID[-1]]

# Joint limits from urdf file
MOVABLE_JOINTS_LIMITS_LOW = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0., 0.]
MOVABLE_JOINTS_LIMITS_HIGH = [2.9671, 1.8326, 2.9671, 0., 2.9671, 3.8223, 2.9671, 0.04, 0.04]

GRIPPER_TARGET_ORIENTATION = [0., 1., 0., 0.]  # Quaternions facing downwards (along -z axis)

TARGET_LINK_ID = 11  # TCP id

# Goal box limits x, y, z
GRIPPER_OFFSET = 0.025
TABLE_PADDING = 0.025
GOAL_X_MIN = 0.4
GOAL_X_MAX = GOAL_X_MIN + 0.2
GOAL_Y_MIN = -0.15
GOAL_Y_MAX = -GOAL_Y_MIN
GOAL_Z_MIN = 0.1 + GRIPPER_OFFSET  # Gripper offset
GOAL_Z_MAX = GOAL_Z_MIN + 0.25

INITIAL_QPOS = [0.85, -0.72, -0.5, -2., -0.33, 1.33, -2., 0., 0.]


def goal_distance(goal_a, goal_b):
    """Distance to goal.

    :param goal_a: (numpy.array) Current / achieved position.
    :param goal_b: (numpy.array) Goal to be reached.

    :return: (float) Distance to goal.
    """
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PandaReachGoalEnv(PyBulletGoalEnv, utils.EzPickle):
    """Robot Arm Panda Reach goal environment.

    Used for HER environments.

    :param distance_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param model_path: (str) Path of simulation model.
    :param initial_pos: (dict) Initial model position, {'joint_name': position}.
    :param sim_time_step: (int) Time step to simulate.
    :param sim_frames_skip: (int) How many frames should be skipped.
    :param sim_n_sub_steps: (int) Sub-steps to be taken.
    :param sim_self_collision: (PyBullet.flag) Collision used in model.
    :param  render: (bool) Should render or not.
    :param render_options: (PyBullet.flag) Render options for PyBullet.
    """

    def __init__(self,
                 distance_threshold=0.05,
                 reward_type='dense',
                 max_steps_per_episode=50,
                 position_gain=1.0,
                 model_path=MODEL_PATH,
                 initial_pos=None,
                 sim_time_step=1.0 / 240.0,
                 sim_frames_skip=0,
                 sim_n_sub_steps=10,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        if initial_pos is None:
            initial_pos = INITIAL_QPOS

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = [position_gain] * len(MOVABLE_JOINTS_ID)
        self.position_gain[-1] = 1.0
        self.position_gain[-2] = 1.0

        # Gripper always closed
        self.gripper_movable = 0
        self.gripper_state = 1

        super(PandaReachGoalEnv, self).__init__(model_path=model_path,
                                                initial_pos=initial_pos,
                                                sim_time_step=sim_time_step,
                                                sim_frames_skip=sim_frames_skip,
                                                sim_n_sub_steps=sim_n_sub_steps,
                                                sim_self_collision=sim_self_collision,
                                                render=render,
                                                render_options=render_options)

        utils.EzPickle.__init__(**locals())

        assert reward_type in ['sparse', 'dense'], 'reward type must be "sparse" or "dense"'
        self.reward_type = reward_type

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.distance_threshold = distance_threshold

    def set_action_space(self):
        """Set action space.

        Action space is 3 dimensional for gripper movement in Cartesian coordinates.
        """
        n_actions = 4
        action_space = spaces.Box(low=-1., high=1., shape=(n_actions,), dtype=np.float32)
        return action_space

    def set_observation_space(self):
        """Set observation space.

        Note: HER style.
        """
        observation = self.get_observation()
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=observation['achieved_goal'].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=observation['achieved_goal'].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=observation['observation'].shape, dtype=np.float32)
        ))

        return observation_space

    def reset_simulation(self):
        """Reset simulation.

        Reset itself is done in parent class.
        Load all necessary models and set start positions.
        """
        # Set gravity and time-step
        self.physics_client.setGravity(0., 0., -9.81)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         useFixedBase=1,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
            self.physics_client.loadURDF(fileName=WORLD_PATH,
                                         useFixedBase=1,
                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         useFixedBase=1)
            self.physics_client.loadURDF(fileName=WORLD_PATH,
                                         useFixedBase=1)

        self.current_episode_steps = 0

        self.set_initial_pos()
        self.goal = self.sample_goal()

        if self.visualize:
            # Load goal sphere(s) for show
            goal = self.goal.copy()
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH,
                                         basePosition=goal.tolist(),
                                         useFixedBase=1)

        observation = self.get_observation()
        return observation

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.

        Note: HER style.
        """
        cur_tcp_pos, cur_tcp_orientation, cur_tcp_lin_vel, cur_tcp_ang_vel = self.get_current_tcp_state()
        cur_object_pos, cur_object_orientation, cur_object_lin_vel, cur_object_ang_vel = self.get_current_object_state()

        cur_object_rel_pos = cur_object_pos - cur_tcp_pos

        observation = np.concatenate([cur_tcp_pos,
                                      cur_tcp_orientation,
                                      cur_tcp_lin_vel,
                                      cur_tcp_ang_vel,
                                      cur_object_rel_pos,
                                      cur_object_pos,
                                      cur_object_orientation,
                                      cur_object_lin_vel,
                                      cur_object_ang_vel,
                                      np.array(self.gripper_state).flatten()])

        return {
            'observation': observation.copy(),
            'achieved_goal': cur_tcp_pos.copy(),
            'desired_goal': self.goal.flatten().copy()
        }

    def step(self, action):
        """Perform (a) simulation step(s).

        Move movable joints within their range (-1, 1).
        Do the actual simulation.
        Calculate the environment stuff (observation, reward, done, info).
        """
        # Clip action values
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cur_tcp_pos, _, _, _ = self.get_current_tcp_state()
        target_pos = cur_tcp_pos + action[:3] * 0.05

        target_pos = np.clip(target_pos,
                             np.array([GOAL_X_MIN, GOAL_Y_MIN, GOAL_Z_MIN]),
                             np.array([GOAL_X_MAX, GOAL_Y_MAX, GOAL_Z_MAX])).tolist()

        joint_ctrl = self.physics_client.calculateInverseKinematics(bodyUniqueId=self.model_id,
                                                                    endEffectorLinkIndex=TARGET_LINK_ID,
                                                                    targetPosition=target_pos,
                                                                    targetOrientation=GRIPPER_TARGET_ORIENTATION)

        joint_ctrl = np.clip(joint_ctrl,
                             np.array(MOVABLE_JOINTS_LIMITS_LOW),
                             np.array(MOVABLE_JOINTS_LIMITS_HIGH))

        self.gripper_state = self.gripper_movable * 0.04 * action[3]
        joint_ctrl[-2] = self.gripper_state
        joint_ctrl[-1] = self.gripper_state
        joint_ctrl = joint_ctrl.tolist()

        self.physics_client.setJointMotorControlArray(bodyUniqueId=self.model_id,
                                                      jointIndices=MOVABLE_JOINTS_ID,
                                                      controlMode=pybullet.POSITION_CONTROL,
                                                      targetPositions=joint_ctrl,
                                                      positionGains=self.position_gain)

        self.do_simulation()

        observation = self.get_observation()
        done = self.is_success(observation['achieved_goal'], observation['desired_goal'])
        info = {'is_success': done}
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

        done = False

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

        self.current_episode_steps += 1

        return observation, reward, done, info

    def set_initial_pos(self):
        """Set initial position."""
        for _ in range(15):
            self.physics_client.setJointMotorControlArray(bodyUniqueId=self.model_id,
                                                          jointIndices=MOVABLE_JOINTS_ID,
                                                          controlMode=pybullet.POSITION_CONTROL,
                                                          targetPositions=self.initial_pos,
                                                          positionGains=[1.0] * len(MOVABLE_JOINTS_ID))

            self.do_simulation()

    def sample_goal(self):
        """Randomly chose goal."""
        x = self.np_random.uniform(GOAL_X_MIN + TABLE_PADDING, GOAL_X_MAX - TABLE_PADDING)
        y = self.np_random.uniform(GOAL_Y_MIN + TABLE_PADDING, GOAL_Y_MAX - TABLE_PADDING)
        if self.np_random.uniform() < 0.5:
            # Target in the air
            z = self.np_random.uniform(GOAL_Z_MIN, GOAL_Z_MAX)
        else:
            z = GOAL_Z_MIN
        goal = np.array([x, y, z])

        return goal.copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current position) and goal.
        """
        distance = goal_distance(achieved_goal, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == 'sparse':
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_object_state(self):
        """Get position and rotation of the block."""
        try:
            object_position, object_orientation = self.physics_client.getBasePositionAndOrientation(self.object_id)
            object_velocity = self.physics_client.getBaseVelocity(self.object_id)
            object_linear_velocity = object_velocity[0]
            object_angular_velocity = object_velocity[1]
        except AttributeError:
            object_position = [0.] * 3
            object_orientation = pybullet.getQuaternionFromEuler([0.] * 3)
            object_linear_velocity = [0.] * 3
            object_angular_velocity = [0.] * 3

        return np.array(object_position).copy(), np.array(object_orientation).copy(), \
               np.array(object_linear_velocity).copy(), np.array(object_angular_velocity).copy()

    def get_current_tcp_state(self):
        """Get position and linear velocity of the tcp."""
        link_state = self.physics_client.getLinkState(self.model_id, TARGET_LINK_ID, computeLinkVelocity=True)

        return np.array(link_state[0]).flatten(), np.array(link_state[1]).flatten(), \
               np.array(link_state[6]).flatten(), np.array(link_state[7]).flatten()

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            # Load goal sphere(s) for show
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=self.goal.tolist(), useFixedBase=1)

            # Camera defaults
            camera_view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[2.25, 0., 1.25],
                                                            cameraTargetPosition=[0., 0., 0.2],
                                                            cameraUpVector=[0., 0., 1.])
            camera_projection_matrix = pybullet.computeProjectionMatrixFOV(fov=45., aspect=1., nearVal=0.1,
                                                                           farVal=3.)

            img = self.physics_client.getCameraImage(width=512, height=512,
                                                     viewMatrix=camera_view_matrix,
                                                     projectionMatrix=camera_projection_matrix)

            return img[2]
        else:
            pass
