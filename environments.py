import numpy as np
# from osim.env import RunEnv
from osim.env import ProstheticsEnv
from gym.spaces import Box, MultiBinary


class RunEnv2(ProstheticsEnv):
    def __init__(self, visualize=False, integrator_accuracy=5e-5, model='2D', prosthetic=False, difficulty=0, skip_frame=3, reward_mult=1.):
        super(RunEnv2, self).__init__(visualize, integrator_accuracy)
        self.args = (model, prosthetic, difficulty)
        self.change_model(*self.args)
        # self.state_transform = state_transform
        # self.observation_space = Box(-1000, 1000, [state_size], dtype=np.float32)
        # self.observation_space = Box(-1000, 1000, [state_transform.state_size], dtype=np.float32)
        self.noutput = self.get_action_space_size()
        self.action_space = MultiBinary(self.noutput)
        self.skip_frame = skip_frame
        self.reward_mult = reward_mult

    def reset(self, difficulty=0, seed=None):
        self.change_model(self.args[0], self.args[1], difficulty, seed)
        s = self.dict_to_vec(super(RunEnv2, self).reset(False))
        # self.state_transform.reset()
        # s = self.state_transform.process(s)
        return s

    def _step(self, action):
        action = np.clip(action, 0, 1)
        info = {'original_reward':0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnv2, self).step(action, False)
            pelvis_X = s['body_pos']['pelvis'][0]
            r = self.x_velocity_reward(s)
            s = self.dict_to_vec(s)  # ndrw subtract pelvis_X
            # s = self.state_transform.process(s)
            info['original_reward'] += r
            reward += r
            if t:
                break
        info['pelvis_X'] = pelvis_X
        return s, reward*self.reward_mult, t, info

    def x_velocity_reward(self, state):
        # if agent is falling, return negative reward
        if state['body_pos']['pelvis'][1] < 0.7:
            return -1 #  -10
        if state['body_pos']['head'][0] < -0.35:
            return -1 #  -10
        # x velocity of pelvis
        # return state['body_vel']['pelvis'][0]
        return state['misc']['mass_center_vel'][0]

    @staticmethod
    def dict_to_vec(dict_):
        """Project a dictionary to a vector.
        Filters fiber forces in muscle dictionary as they are already
        contained in the forces dictionary.
        """
        # length without prosthesis: 443 (+ 22 redundant values)
        # length with prosthesis: 390 (+ 19 redundant values)

        # np.array([val_or_key if name != 'muscles'
        #           else list_or_dict[val_or_key]
        #           for name, subdict in dict_.items()
        #           for list_or_dict in subdict.values()
        #           for val_or_key in list_or_dict
        #           if val_or_key != 'fiber_force'])

        #         return np.array([val_or_key if name != 'muscles'
        #                 else list_or_dict[val_or_key]
        #                 for name, subdict in dict_.items()
        #                 for list_or_dict in subdict.values()
        #                 for val_or_key in list_or_dict
        #                 if val_or_key != 'fiber_force'])

        # projection = np.array([])
        # pelvis_X = dict_['body_pos']['pelvis'][0]
        # for dict_name in ['joint_pos', 'joint_vel', 'body_pos', 'body_vel', 'body_pos_rot', 'body_vel_rot', 'misc']:
        #     for dict_name_2 in dict_[dict_name]:
        #         a = dict_[dict_name][dict_name_2]
        #         if len(a) > 0:
        #             if dict_name == 'body_pos':
        #                 a[0] -= pelvis_X
        #             projection = np.concatenate((projection, np.array(a)))
        # assert len(projection) == 196

        projection = np.array([])
        pelvis_X = dict_['body_pos']['pelvis'][0]
        for dict_name in ['joint_pos', 'joint_vel', 'body_pos', 'body_vel', 'body_pos_rot', 'body_vel_rot', 'misc']:
            for dict_name_2 in dict_[dict_name]:
                a = dict_[dict_name][dict_name_2]
                if len(a) > 0:
                    if dict_name == 'body_pos':
                        a[0] -= pelvis_X
                    projection = np.concatenate((projection, np.array(a)))
        for dict_name_m in dict_['muscles']:
            for dict_name_m2 in ['activation', 'fiber_length', 'fiber_velocity']:
                a = [dict_['muscles'][dict_name_m][dict_name_m2]]
                projection = np.concatenate((projection, np.array(a)))
        assert len(projection) == 262

        return projection
