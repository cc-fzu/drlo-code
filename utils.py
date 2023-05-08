import numpy as np
import pandas as pd
from math import cos, asin, sqrt, exp



# 经纬度距离计算
def cal_distance(lat_a, lng_a, lat_b, lng_b):
    '''
    lng_a: 经度A
    lat_a: 纬度A
    lng_b: 经度B
    lat_b: 纬度B
    return距离(km)
    '''
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - cos((lat_b - lat_a) * p) / 2 + cos(lat_a * p) * cos(lat_b * p) * (1 - cos((lng_b - lng_a) * p)) / 2
    return 12742 * asin(sqrt(a))   # 2*R*asin...

# 将网络输出的动作转换为0/1形式
def transform_action(output, num):
    output = np.array(output)
    output = output.ravel()
    index = []
    # output_list = []
    for i in range(num):
        max_index = 0
        max = -999999
        for j in range(len(output)):
            if j not in index and output[j] > max:
                max_index = j
                max = output[j]
        index.append(max_index)

    for i in range(len(output)):
        if i not in index:
            output[i] = 0
        else:
            output[i] = 1
    return output

class Assign:
    def __init__(self, bs_num, es_num, data_path, days, init_vector):
        self.init_vector = init_vector
        # self.init_vector = server_list_init
        self.bs_num = bs_num
        self.es_num = es_num
        self.data_path = data_path
        self.days = days
        self.data = pd.read_csv(self.data_path, header=None, nrows=self.bs_num)
        # self.data[5], self.data[6] = self.data[1].str.split('/', 1).str
        # s = self.data[1].str.split('/', 1)
        # self.data[4], self.data[5] = s.str[0], s.str[1]
        self.data_df = pd.DataFrame(self.data)
        self.data = self.data.values.tolist()
        self.workload_list = []

    def get_init_action(self):
        action_init = self.init_vector
        action = np.zeros(self.bs_num)
        for i in action_init:
            action[i] = 1
        return action

    def get_es_label(self, action):
        # 分配基站
        server_list = []
        labels = []
        for i, a in enumerate(action):
            if a == 1:
            # if a == 0:
                server_list.append(i)
        for bs_index in range(self.bs_num):
            min_dist = 9999999999
            es_index = -1
            for server_index in server_list:
                dist = cal_distance(float(self.data[bs_index][1]), float(self.data[bs_index][2]),
                                     float(self.data[server_index][1]), float(self.data[server_index][2]))
                if dist < min_dist:
                    min_dist = dist
                    es_index = server_index
            labels.append(es_index)

        return labels


    # 平均访问时延
    def get_aver_delay(self, labels, day_i, data_j):
        # 计算时延

        delay_list = []
        for i, es_index in enumerate(labels):
            delay = cal_distance(float(self.data[i][1]), float(self.data[i][2]),
                                      float(self.data[es_index][1]), float(self.data[es_index][2]))
            delay_list += [delay] * int(self.data[i][4+data_j+2*day_i]) #访问时延乘用户数
        delay_array = np.array(delay_list)  #每个用户数的时延
        # print("delay_list:")
        # print(delay_array)
        aver_delay = np.mean(delay_array)

        return aver_delay

    # 计算状态 - 每天的最大访问延迟
    def get_delay_list(self, action, data_j):
        delay_list = []
        labels = self.get_es_label(action)
        for day_i in range(self.days):
            aver_delay = self.get_aver_delay(labels, day_i, data_j)
            delay_list.append(aver_delay)

        delay_arr = np.array(delay_list)

        delay_max = delay_arr.max()
        # max_day = delay_arr.argmax()
        delay_min = delay_arr.min()
        # min_day = delay_arr.argmin()


        d_max_ = delay_arr.max()
        d_min_ = delay_arr.min()

        return delay_arr, delay_max, delay_min, d_max_, d_min_

    # # 计算工作负载
    def get_workload_bias(self, labels, day_i, data_j):
        workload_list = [0] * self.bs_num
        for i, es_index in enumerate(labels):
            workload_list[es_index] += self.data[i][3+data_j+2*day_i]

        workload_es = []
        for w in workload_list:
            if w != 0:
                workload_es.append(w)
        workload_es = np.array(workload_es)
        aver_w = np.mean(workload_es)
        w_bias = np.sqrt(((workload_es - aver_w) ** 2).sum() / self.es_num)

        return w_bias

    def get_w_bias_list(self, action, data_j):
        w_bias_list = []
        labels = self.get_es_label(action)
        for day_i in range(self.days):
            w_bias = self.get_workload_bias(labels, day_i, data_j)
            w_bias_list.append(w_bias)

        w_bias_arr = np.array(w_bias_list)
        w_bias_max = w_bias_arr.max()

        return w_bias_max


    def get_init_state(self, data_j):
        init_action = self.get_init_action()
        init_state, init_delay_max, init_delay_min, d_max_, d_min_ = self.get_delay_list(init_action, data_j)
        init_w_bias_max = self.get_w_bias_list(init_action, data_j)
        return init_state, init_delay_max, init_delay_min, init_w_bias_max


    def get_reward(self, delay_max, init_delay_min):
        print("d_max:", delay_max, "init_d_min", init_delay_min)
        if delay_max > init_delay_min:
            reward = -exp(delay_max/init_delay_min - 1) + 1
        elif delay_max < init_delay_min:
            reward = exp(init_delay_min/delay_max - 1) - 1
        else:
            reward = 0
        return reward

    def next_step(self, action, init_delay_min, data_j):
        new_state, delay_max, delay_min, d_max_, d_min_ = self.get_delay_list(action, data_j)
        reward = self.get_reward(delay_max, init_delay_min)
        w_bias_max = self.get_w_bias_list(action, data_j)

        return new_state, delay_max, delay_min, reward, w_bias_max

