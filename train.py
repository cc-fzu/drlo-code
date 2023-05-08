import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import Actor, Critic, OUNoise, ReplayBuffer
from Environment import Env
import utils
import pandas as pd
import json
import random
import sys
from sklearn import preprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

np.set_printoptions(threshold=20)
# np.set_printoptions(threshold=sys.maxsize)
data_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))


def ddpg_play(ddpg_config):
    tf.keras.backend.set_floatx('float64')

    mulswi = 30  # 文件夹名称

    # 相关参数定义
    BUFFER_SIZE = 40000   # 缓冲池的大小

    BATCH_SIZE = 256     # batch_size的大小
    # BATCH_SIZE = 512     # batch_size的大小
    # BATCH_SIZE = 128  # batch_size的大小

    GAMMA = 0.9  # 折扣系数
    TAU = 0.001  # target网络软更新的速度
    LR_A = 5e-7  # Actor网络的学习率
    LR_C = 5e-6  # Critic网络的学习率

    # 变量设置
    episode = 2000  # 迭代的次数
    max_step = 30
    char_num = 80
    explore = 0.9
    epsilon_rate = 0.0005
    EXPLORE = explore * episode


    # 加载环境
    # loc_path = 'dataset/data_6.15-8.15/loc_6.15-8.15.csv'
    # 时间段
    # time = '6.16-8.15'
    time = '6.16-11.30'

    # 设定服务器放置数量
    es_ratio = ddpg_config['es_ratio']  #服务器部署率
    bs_num = ddpg_config['bs_num']
    es_num = int(bs_num * es_ratio)

    es_r = str(bs_num) + "_" + str(es_ratio)

    # 获取初始解服务器向量
    init_a_dict = {"3000_0.04": 'kmeans', "3000_0.05": 'kmeans', "3000_0.06": 'kmeans',
                   "3000_0.07": 'kmeans', "3000_0.08": 'kmeans', "3000_0.09": 'kmeans',
                   "3000_0.1": 'kmeans', "3000_0.11": 'kmeans', "3000_0.12": 'topk',
                   "3000_0.13": 'hap', "3000_0.14": 'topk', "3000_0.15": 'topk',
                   "3000_0.16": 'topk', "3000_0.17": 'topk', "3000_0.18": 'topk',
                   "3000_0.19": 'topk', "3000_0.2": 'topk', "500_0.1":'kmeans'}
    init_action = init_a_dict[es_r]

    init_action_file = 'test_img/' + time + '/compare_result/' + init_action + '_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    init_result = pd.read_csv(init_action_file, header=None)
    init_result = init_result.values.tolist()
    init_result = np.array(init_result)
    init_result = init_result.flatten()
    init_vector = []
    for i in range(bs_num):
        if init_result[i] not in init_vector:
            init_vector.append(init_result[i])

    action_y = ddpg_config['tanh_y']

    # 训练集
    # train_days = 46
    train_days = 31
    days = 31

    traindata_num = train_days - days + 1
    train_dir = 'dataset/data_' + time + r'/traindata'
    # train_data = train_dir +'/data_6.16-7.31_days.csv'
    # train_data = train_dir +'/data_7.1-7.31_days.csv'
    train_data = train_dir +'/data_7.1-7.31_w+n_' + str(bs_num) + 'bs.csv'

    total_step = 0  # 总共运行了多少步
    state_dim = days
    action_dim = bs_num

    # flag = True  # 判断模型是否训练到最佳

    # 可视化集合定义
    reward_list = []  # 记录所有的rewards进行可视化展示
    loss_list = []  # 记录损失函数进行可视化展示
    delay_list = []
    workload_list = []
    ep_list = []
    # ou_noise = OUNoise.OU(processes=action_dim)

    # 神经网络相关操作定义
    buff = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)  # 创建缓冲区
    # 创建四个神经网络
    actor = Actor.Actor(state_dim, action_dim, LR_A)
    actor_target = Actor.Actor(state_dim, action_dim, LR_A)
    critic = Critic.Critic(state_dim, action_dim, LR_C)
    critic_target = Critic.Critic(state_dim, action_dim, LR_C)

    # 给target网络设置参数
    actor_weight = actor.model.get_weights()
    critic_weight = critic.model.get_weights()
    actor_target.model.set_weights(actor_weight)
    critic_target.model.set_weights(critic_weight)

    # Now load the weight
    # print("Now we load the weight")
    # 保存参数模型
    actor_model_path = 'src/actormodel_' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r.h5'
    critic_model_path = 'src/criticmodel_' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r.h5'

    # try:
    #     actor.model.load_model_weights(actor_model_path)
    #     critic.model.load_weights(critic_model_path)
    #     actor.target_model.load_weights(actor_model_path)
    #     critic.target_model.load_weights(critic_model_path)
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    # actor.model.summary()
    #开始
    print("Experiment Start.")

    data_path = train_data
    env = Env(bs_num, es_num, data_path, days, init_vector)
    learning_rate_A = LR_A
    learning_rate_C = LR_C

    # flag = 0
    for ep in range(episode):
        lastep_max_r = -999.9
        for j in range(traindata_num):
            state, init_delay_max, init_delay_min, init_w_bias_max = env.reset(j)
            print("-" * char_num)
            print("Episode: " + str(ep) + "Replay Buffer" + str(buff.getCount()))
            print("第" + str(j) + "个训练集初始最小延迟init_Delay_min: " + str(init_delay_min))
            print("第" + str(j) + "个训练集初始最大延迟init_Delay_max: " + str(init_delay_max))
            print("第" + str(j) + "个训练集初始最大工作负载标准差init_W_Bias_max: " + str(init_w_bias_max))
            print("初始状态：" + str(state))
            total_reward = 0.0
            total_loss = 0.0
            total_delay = 0.0
            total_workload = 0.0
            step = 0

            for t in range(max_step):

                state = state.reshape(1, state_dim)
                print("state: ", state)

                action_origin = actor.model.predict(state) #做出动作
                action = action_origin[0]

                print("action_sum: " + str(action.sum()))
                print("ab_action_sum: " + str(abs(action).sum()))
                print("action: ", action)

                if ep <= EXPLORE:

                    epsilon = abs(abs(action).sum()) * epsilon_rate
                    noise = np.random.normal(0, 0.5, action_dim)
                    noise = noise * epsilon

                    action = np.where((action+noise > -action_y)&(action+noise < action_y), action+noise, action-noise)

                    # action = action.clip(min = -action_y, max = action_y)

                    print("epsilon:", epsilon)


                next_state, delay_max, delay_min, reward, done, w_bias_max = env.step(action, j)
                next_state = next_state.reshape(state_dim, )
                print("next_state: ", next_state)

                if t == max_step - 1:
                    done = True

                buff.add(state.squeeze(), action, reward, next_state, done)

                # 采样进行更新
                states, actions, rewards, next_states, dones = buff.getBatch(BATCH_SIZE)

                target_q_values = critic_target.model.predict([next_states, actor_target.model.predict(next_states)])
                yi = compute_yi(rewards, target_q_values, dones, GAMMA)
                loss = critic.train(states, actions, yi, learning_rate_C)

                with open(str(mulswi) + '/lossLog' + time + '_' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) +'bs_' + str(es_ratio) + 'r.txt', 'a') as file:
                    file.write(str(loss) + '\n')

                a_for_grads = actor.model.predict(states)
                a_grads = critic.q_grads(states, a_for_grads)
                actor.train(states, a_grads, learning_rate_A)
                target_update(actor, actor_target, critic, critic_target, TAU)  # 网络更新

                print("-" * char_num)
                print("Episode", ep, "Step", step, "Action", utils.transform_action(action, env.es_num), "DelayMax", delay_max, "W_Bias_max", w_bias_max, "Reward", reward, "Loss", np.array(loss))


                # if (ep == episode - 1) and (t == max_step - 1):
                #     action_place = utils.transform_action(action, es_num)
                #     es_place_file = 'es_place/place_' + time + '_' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' +  str(es_ratio) + 'r.csv'
                #     action_df = pd.DataFrame(action_place.T)
                #     action_df.to_csv(es_place_file, index=False, header=False)

                # 选取最大奖励的放置位置
                if ep > EXPLORE:
                    if reward > lastep_max_r:
                        lastep_max_r = reward
                        action_final = action
                if (ep == episode - 1) and (t == max_step - 1):
                    action_place = utils.transform_action(action_final, es_num)
                    es_place_file = 'es_place/place_' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' +  str(es_ratio) + 'r.csv'
                    action_df = pd.DataFrame(action_place.T)
                    action_df.to_csv(es_place_file, index=False, header=False)
                    print("")
                    print("*" * char_num)
                    print("Successfully save action_file! last step max_reward = " + str(lastep_max_r))
                    print("*" * char_num)

                total_reward += reward
                total_loss += np.array(loss)
                total_delay += delay_max
                total_workload += w_bias_max

                step += 1
                total_step += 1
                state = next_state

            if ep % 5 == 0:
                print("Now we save model")
                actor.model.save_weights(actor_model_path, overwrite=True)
                critic.model.save_weights(critic_model_path, overwrite=True)

        # 打印相关信息
        print("")
        print("=" * char_num)
        print("TOTAL REWARD @ " + str(ep) + "-th Episode  : Reward " + str(total_reward / step))
        print("TOTAL LOSS @ " + str(ep) + "-th Episode  : LOSS " + str(total_loss / step))
        print("TOTAL Delay Max @ " + str(ep) + "-th Episode  : Delay_Max " + str(total_delay / step))
        print("TOTAL W_Bias Max @ " + str(ep) + "-th Episode  : W_Bias_Max " + str(total_workload / step))
        print("TOTAL REBUFF SIZE: " + str(buff.getCount()))
        print("=" * char_num)
        print("")


        if total_reward/step > 0 and ep > EXPLORE:
            learning_rate_A = learning_rate_A * 0.9 ** (ep / 100)
            # learning_rate_C = learning_rate_C * 0.9 ** (ep / 100)



        if j == traindata_num - 1 and ep % 5 == 0:
            print("ep:", ep, "data_num:", j)
            print("Save img")
            # 绘制图像，并保存
            aver_reward = total_reward / step
            aver_loss = total_loss / step
            aver_delay = total_delay / step
            aver_workload = total_workload / step

            reward_list.append(aver_reward)
            loss_list.append(aver_loss)
            delay_list.append(aver_delay)
            workload_list.append(aver_workload)
            ep_list.append(ep)

            #保存记录到.txt
            f_r = open('train_log/' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r_log.txt', mode='a+')
            f_r.write("episode:" + str(ep)+ ",loss:"+str(aver_loss)+ ",delay:"+str(aver_delay)+
                      ",reward:"+str(aver_reward)+ ",workload:"+ str(aver_workload) + '\n')


            plt.cla()  # 清除
            plt.plot(ep_list, loss_list)
            plt.xlabel("Episode")
            plt.ylabel("Training Loss")
            # plt.title("loss-ep")
            img_name = 'img/loss/' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r_loss.png'
            plt.savefig(img_name)

            plt.cla()  # 清除
            plt.plot(ep_list, reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            # plt.title("loss-ep")
            img_name = 'img/reward/' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r_reward.png'
            plt.savefig(img_name)

            plt.cla()  # 清除
            plt.plot(ep_list, delay_list)
            plt.xlabel("Episode")
            plt.ylabel("Delay (km)")
            # plt.title("delay-ep")
            img_name = 'img/delay/' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' +  str(es_ratio) + 'r_delay.png'
            plt.savefig(img_name)

            plt.cla()  # 清除
            plt.plot(ep_list, workload_list)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            plt.xlabel("Episode")
            plt.ylabel("Workload Balancing (min)")
            # plt.title("delay-ep")
            img_name = 'img/workload_bias/' + str(episode) + 'iters_' + str(max_step) + 'steps_' + str(bs_num) + 'bs_' + str(es_ratio) + 'r_workload_bias.png'
            plt.savefig(img_name)

def softmax_(inputs):
    x_max = tf.reduce_max(inputs)

    return np.exp(inputs-x_max) / np.sum(np.exp(inputs-x_max))

def compute_yi(rewards, target_q_values, dones, GAMMA):
    yi = np.asarray(target_q_values)
    for i in range(target_q_values.shape[0]):
        if dones[i]:
            yi[i] = rewards[i]
        else:
            yi[i] = GAMMA * target_q_values[i] + rewards[i]
    return yi

def target_update(actor, actor_target, critic, critic_target, tau):
    actor_weights = actor.model.get_weights()
    t_actor_weights = actor_target.model.get_weights()
    critic_weights = critic.model.get_weights()
    t_critic_weights = critic_target.model.get_weights()

    for i in range(len(actor_weights)):
        t_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * t_actor_weights[i]

    for i in range(len(critic_weights)):
        t_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * t_critic_weights[i]

    actor_target.model.set_weights(t_actor_weights)
    critic_target.model.set_weights(t_critic_weights)

    # Gaussian噪声
def gaussian(mu, sigma, processes):
    mean = mu
    sigma_ = sigma
    processes_ = processes
    # random.gauss(mean, sigma_)
    return np.random.normal(mean, sigma_, processes_)

if __name__ == "__main__":
   with open('parameter.json') as jconfig:
       esplace_config = json.load(jconfig)
   ddpg_play(esplace_config)






