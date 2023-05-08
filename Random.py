import random
import pandas as pd
from math import  cos, asin, sqrt
# random.seed(736)# 确定随机数种子

class RandomServerPlacement:
    def __init__(self, bs_num, es_num, data, server_list):
        self.bs_num = bs_num  # 基站数
        self.es_num = es_num  # 服务器数
        self.data = pd.read_csv(data, header=None, nrows=self.bs_num)
        s = self.data[1].str.split('/', 1)
        self.data[4], self.data[5] = s.str[0], s.str[1]
        # 基站位置数据转换为列表
        self.data = self.data.values.tolist()
        self.server_list = server_list

        # 计算距离
    def cal_distance(self, lat_a, lng_a, lat_b, lng_b):
        '''
        lng_a: 经度A
        lat_a: 纬度A
        lng_b: 经度B
        lat_b: 纬度B
        return距离(km)
        '''
        p = 0.017453292519943295  # Pi/180
        a = 0.5 - cos((lat_b - lat_a) * p) / 2 + cos(lat_a * p) * cos(lat_b * p) * (
                    1 - cos((lng_b - lng_a) * p)) / 2
        return 12742 * asin(sqrt(a))  # 2*R*asin...

    def random_place(self):
        labels = []
        for bs_index in range(self.bs_num):
            min_dist = 999999
            es_index = -1
            for server_index in self.server_list:
                dist = self.cal_distance(float(self.data[bs_index][4]), float(self.data[bs_index][5]),
                                         float(self.data[server_index][4]), float(self.data[server_index][5]))
                if dist < min_dist:
                    min_dist = dist
                    es_index = server_index
            labels.append(es_index)
        # print(labels)
        return labels

#
# time = '6.15-8.15'
# # 基站信息文件
# data_dir = 'dataset/data_' + time + r'/traindata'
# data_file = data_dir + '/data_6.16-6.30.csv'
# server_num = sum(1 for line in open(data_file))
# bs_num = 300
# iters = 10

# random_ = RandomServerPlacement(server_num, bs_num, data_file)
# random_result = random_.random_place()