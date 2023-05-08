import utils
import numpy as np

class Env:
    def __init__(self, bs_num, es_num, data_path, days, init_vector):
        self.bs_num = bs_num
        self.es_num = es_num
        self.data_path = data_path
        self.days = days
        self.init_vector = init_vector
        self.Assign = utils.Assign(self.bs_num, self.es_num, self.data_path, self.days, self.init_vector)
        self.init_delay_max = 0
        self.init_delay_min = 0

    def reset(self, data_j):
        init_state, init_delay_max, init_delay_min, init_w_bias_max = self.Assign.get_init_state(data_j)
        self.init_delay_max = init_delay_max
        self.init_delay_min = init_delay_min

        return init_state, init_delay_max, init_delay_min, init_w_bias_max

    def step(self, action, data_j):
        action = utils.transform_action(action, self.es_num)
        new_state, delay_max, delay_min, reward, w_bias_max = self.Assign.next_step(action, self.init_delay_min, data_j)
        # new_state, delay_max, delay_min, reward = self.Assign.next_step(action, self.init_delay_max, data_j)

        return new_state, delay_max, delay_min, reward, False, w_bias_max


