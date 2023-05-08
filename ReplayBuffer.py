# 导入相关模块
from collections import deque
import random
import numpy as np

# 建立经验缓冲池
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # 经验池最大大小
        self.num_experiences = 0    # 当前经验池的中经验的数量
        self.buffer = deque()       # 经验池
        # self.buffer = []
        # self.min_reward = 999999
        # self.min_index = -1
        # self.index = -1
    # 获取batsize的历史经验
    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return self.sample(self.num_experiences) # 当经验池不足batch_size时,全部返回
        else:
            return self.sample(batch_size)   # 当经验池充足时，随机返回batch_size的经验

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        actions = np.array(actions).reshape(batch_size, -1)
        rewards = np.array(rewards).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done


    # 判断缓冲区是否满了
    def is_full(self):
        if self.buffer_size == self.num_experiences:
            return True
        else:
            return False

    # 返回缓冲池的大小
    def getSize(self):
        return self.buffer_size

    # 给缓冲池添加内容
    def add(self, state, action, reward, new_state, done):
        random.shuffle(self.buffer)
        experience = (state, action, reward, new_state, done)
        # experience = [state, action, reward, new_state, done]
        # print(experience)
        if self.num_experiences < self.buffer_size:
            # self.index += 1
            # if reward < self.min_reward:
            #     self.min_reward = reward
            #     self.min_index = self.index
            # 如果缓冲区没有存满，则直接进行存储
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            # 如果缓冲区存满了，删除奖励最小的经验
            # self.buffer = deque(sorted(self.buffer, key=lambda temp: temp[2]))
            self.buffer = deque(sorted(self.buffer, key=lambda temp: temp[2]))
            # self.buffer.popleft()
            # self.buffer.append(experience)
            experience_min = self.buffer.popleft()

            # print("最小奖励：" + str(experience_min[2]))
            if experience_min[2] < experience[2]:
            # if experience_max[2] > experience[2]:
                self.buffer.append(experience)
            else:
                self.buffer.append(experience_min)


            # experience_min = (deque(sorted(self.buffer, key=lambda temp: temp[2]))).popleft()
            # print(experience_min)
            # reward_min = experience_min[2]
            # if reward_min < experience[2]:
            #     self.buffer.remove(experience_min)
                # self.buffer.index(experience_min)
                # self.buffer.append(experience)

    # 返回当前存储经验的大小
    def getCount(self):
        return self.num_experiences

    # 重置缓冲区
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    # 将缓冲区存储成为文件
    def saveBuffer(self):
        file_path = "src\\buffer"
        with open(file_path, "w") as f:
            for item in self.buffer:
                s = str(item[0]) + str(item[1]) + str(item[2]) + str(item[3]) + str(item[4])
                f.write(s)