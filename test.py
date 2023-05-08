import pandas as pd
import csv
from math import cos, asin, sqrt
import matplotlib.pyplot as plt
import json
from pylab import xticks, np
from matplotlib.pyplot import MultipleLocator

'''
每日访问时延和工作负载均衡情况变化曲线图绘制
'''


# 测试：根据位置放置基站，计算时延和工作负载
class test_utils:
    def __init__(self, bs_num, es_num, data, result):
        self.bs_num = bs_num
        self.es_num = es_num
        self.data = pd.read_csv(data, header=None, nrows=bs_num)
        s = self.data[0].str.split('/', 1)
        self.data[3], self.data[4] = s.str[0], s.str[1]
        # 基站位置数据转换为列表
        self.data = self.data.values.tolist()
        self.result = result    #最终所有基站所属服务器的索引

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

    # 计算平均访问时延：
    def aver_delay(self):
        delay_list = []
        for i, es_index in enumerate(self.result):
            delay = self.cal_distance(float(self.data[i][3]), float(self.data[i][4]),
                                      float(self.data[es_index][3]), float(self.data[es_index][4]))
            delay_list += [delay] * self.data[i][2]
        delay_array = np.array(delay_list)
        aver_delay = np.mean(delay_array)
        return aver_delay
    # 计算工作负载标准差
    def workload_bias(self):
        c = {} #簇
        for i, es_index in enumerate(self.result):
            if es_index not in c:
                c[es_index] = self.data[i][1]
            else:
                c[es_index] += self.data[i][1]
        workload = []
        for workload_i in c.values():
            workload.append(workload_i)
        workload = np.array(workload)
        aver_workload = np.mean(workload)
        workload_bias = np.sqrt(((workload - aver_workload) ** 2).sum() / self.es_num)

        return workload_bias

# 开始测试
with open('parameter.json') as jconfig:
   esplace_config = json.load(jconfig)

es_ratio = esplace_config['es_ratio']   # 服务器部署率
bs_num = esplace_config['bs_num']       # 基站数
es_num = int(bs_num * es_ratio)              # 服务器数

time = '6.16-11.30'


# 循环，获取每天基站数据 - 计算访问时延和工作负载
def test_dynamic(test_data, kmeans_result, topk_result, random_result, ddpg_result, ga_result):
    kmeans_d = []
    topk_d = []
    random_d = []
    ddpg_d = []
    ga_d = []

    kmeans_w = []
    topk_w = []
    random_w = []
    ddpg_w = []
    ga_w = []

    for testdata_day in test_data:
        kmeans_utils = test_utils(bs_num, es_num, testdata_day, kmeans_result)
        kmeans_delay = kmeans_utils.aver_delay()
        kmeans_workload_bias = kmeans_utils.workload_bias()
        kmeans_d.append(kmeans_delay)
        kmeans_w.append(kmeans_workload_bias)

        topk_utils = test_utils(bs_num, es_num, testdata_day, topk_result)
        topk_delay = topk_utils.aver_delay()
        topk_workload_bias = topk_utils.workload_bias()
        topk_d.append(topk_delay)
        topk_w.append(topk_workload_bias)

        random_utils = test_utils(bs_num, es_num, testdata_day, random_result)
        random_delay = random_utils.aver_delay()
        random_workload_bias = random_utils.workload_bias()
        random_d.append(random_delay)
        random_w.append(random_workload_bias)

        ddpg_utils = test_utils(bs_num, es_num, testdata_day, ddpg_result)
        ddpg_delay = ddpg_utils.aver_delay()
        ddpg_workload_bias = ddpg_utils.workload_bias()
        ddpg_d.append(ddpg_delay)
        ddpg_w.append(ddpg_workload_bias)


        ga_utils = test_utils(bs_num, es_num, testdata_day, ga_result)
        ga_delay = ga_utils.aver_delay()
        ga_workload_bias = ga_utils.workload_bias()
        ga_d.append(ga_delay)
        ga_w.append(ga_workload_bias)

    return kmeans_d, topk_d, random_d, ddpg_d, ga_d, kmeans_w, topk_w, random_w, ddpg_w, ga_w

def get_aver_d(kmeans_d, topk_d, random_d, ddpg_d, ga_d):
    '''
    求三十天访问时延平均值
    '''

    # 访问时延
    aver_kmeans_d = np.mean(kmeans_d)
    aver_topk_d = np.mean(topk_d)
    aver_random_d = np.mean(random_d)
    aver_ddpg_d = np.mean(ddpg_d)
    aver_ga_d = np.mean(ga_d)

    return aver_kmeans_d, aver_topk_d, aver_random_d, aver_ddpg_d, aver_ga_d

def get_aver_w(kmeans_w, topk_w, random_w, ddpg_w, ga_w):
    '''
    求三十天工作负载平均值
    '''
    aver_kmeans_w = np.mean(kmeans_w)
    aver_topk_w = np.mean(topk_w)
    aver_random_w = np.mean(random_w)
    aver_ddpg_w = np.mean(ddpg_w)
    aver_ga_w= np.mean(ga_w)

    return aver_kmeans_w, aver_topk_w, aver_random_w, aver_ddpg_w, aver_ga_w


if __name__ == "__main__":
    '''
    获取动态变化图，并将求出的平均结果存入csv文件
    '''

    # 读取各方法基站放置位置
    file_kmeans = 'compares/kmeans_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    kmeans_result = pd.read_csv(file_kmeans, header=None)
    kmeans_result = kmeans_result.values.tolist()
    kmeans_result = np.array(kmeans_result)
    kmeans_result = kmeans_result.flatten()

    file_topk = 'compares/compare_result/topk_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    topk_result = pd.read_csv(file_topk, header=None)
    topk_result = topk_result.values.tolist()
    topk_result = np.array(topk_result)
    topk_result = topk_result.flatten()

    file_random = 'compares/compare_result/random_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    random_result = pd.read_csv(file_random, header=None)
    random_result = random_result.values.tolist()
    random_result = np.array(random_result)
    random_result = random_result.flatten()

    file_ddpg = 'compares/compare_result/ddpg_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    ddpg_result = pd.read_csv(file_ddpg, header=None)
    ddpg_result = ddpg_result.values.tolist()
    ddpg_result = np.array(ddpg_result)
    ddpg_result = ddpg_result.flatten()

    file_ga = 'compares/compare_result/ga_' + str(bs_num) + '_' + str(es_ratio) + '.csv'
    ga_result = pd.read_csv(file_ga, header=None)
    ga_result = ga_result.values.tolist()
    ga_result = np.array(ga_result)
    ga_result = ga_result.flatten()


    time_month = '8.1-8.31'
    begin_day = 1
    days = 31
    test_data = []
    for i in range(days):
        if i < 9:
            test_data.append('dataset/data_' + time + r'/testdata_day/data_2014080' + str(i + begin_day) + '_' + str(bs_num) + 'bs.csv')
        else:
            test_data.append('dataset/data_' + time + r'/testdata_day/data_201408' + str(i + begin_day) + '_' + str(bs_num) + 'bs.csv')

    # 添加点，绘制图
    day = [i+begin_day for i in range(days)]    #横坐标 - 日期 by day
    # print(day)
    kmeans_d, topk_d, random_d, ddpg_d, ga_d, kmeans_w, topk_w, random_w, ddpg_w, ga_w = test_dynamic(test_data, kmeans_result, topk_result, random_result, ddpg_result, ga_result)

    aver_kmeans_d, aver_topk_d, aver_random_d, aver_ddpg_d, aver_ga_d = get_aver_d(kmeans_d, topk_d, random_d, ddpg_d, ga_d)
    aver_kmeans_w, aver_topk_w, aver_random_w, aver_ddpg_w, aver_ga_w = get_aver_w(kmeans_w, topk_w, random_w, ddpg_w, ga_w)

    '''
    存入所有平均数值
    '''
    aver_data = 'test_img/' + time + '/average_' + time_month + '.csv'
    aver_data_pd = pd.read_csv(aver_data, header=None)
    aver_data_list = aver_data_pd.values.tolist()

    if [str(bs_num), str(es_ratio),
        str(aver_kmeans_d), str(aver_topk_d), str(aver_random_d), str(aver_ddpg_d), str(aver_ga_d),
        str(aver_kmeans_w), str(aver_topk_w), str(aver_random_w), str(aver_ddpg_w), str(aver_ga_w)] not in aver_data_list:
        with open(aver_data, "a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([bs_num, es_ratio,
                             aver_kmeans_d, aver_topk_d, aver_random_d, aver_ddpg_d, aver_ga_d,
                             aver_kmeans_w, aver_topk_w, aver_random_w, aver_ddpg_w, aver_ga_w])


    # 不同基站和部署率纵坐标范围设置 - 前两个延迟图，后两个负载图
    lim_dict = {"500_0.1": [2.4, 4.95, 0.0, 1.35e5], "1000_0.1": [1.7, 3.4, 0.0, 1.7e5], "1500_0.1": [1.6, 2.62, 0.0, 1.7e5],
                "2000_0.1": [1.29, 2.399, 0.0, 1.6e5], "2500_0.1": [1.08, 2.05, 0.0, 1.5e5], "3000_0.1": [1.06, 1.79, 0.0, 1.5e5],
                "3000_0.04": [1.78, 3.2, 0.0, 3.7e5], "3000_0.05": [1.6, 2.75, 0.0, 2.8e5], "3000_0.06": [1.45, 2.35, 0.0, 2.3e5],
                "3000_0.07": [1.35, 2.18, 0.0, 2.05e5], "3000_0.08": [1.2, 1.96, 0.0, 1.85e5], "3000_0.09": [1.15, 1.85, 0.0, 1.6e5],
                "3000_0.11": [1.0, 1.62, 0.0, 1.4e5], "3000_0.12": [0.88, 1.54, 0.0, 1.3e5], "3000_0.13": [0.82, 1.52, 0.0, 1.2e5],
                "3000_0.14": [0.76, 1.44, 0.0, 1.15e5]
                }

    es_r = str(bs_num) + "_" + str(es_ratio)
    lim_list = lim_dict[es_r]
    # 图像参数设置
    linewidth_d = 4.3
    fontsize_d = 28.3
    legendsize_d = 28.3
    markersize_ = 12.5
    markeredge_ = 1

    # 创建画布
    # plt.figure()
    plt.cla()  # 清除
    plt.figure(figsize=(13, 6))  # 调整图片尺寸


    '''
    访问时延图 - 固定基站数和部署率 - 动态 - 随时间变化
    '''

    plt.xlim(0.4, 31.6)
    plt.ylim(lim_list[0], lim_list[1])
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1) #设置绘图区和画布的边距
    xticks(np.arange(1, 32, 3))
    # my_xticks=np.arange(1, 32, 2)
    # my_yticks=np.arange(lim_list[0], lim_list[1], 0.4) # 控制间隔

    # plt.xticks(my_xticks)
    # plt.yticks(my_yticks)

    plt.xlabel("Days of August", fontsize=fontsize_d, labelpad=8.5)
    plt.ylabel("Average Delay (km)", fontsize=fontsize_d, labelpad=7)

    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['left'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['top'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['right'].set_linewidth(2)  # 设置坐标轴粗细

    y_major_locator = MultipleLocator(0.3)  # 设置y坐标轴间隔
    ax.yaxis.set_major_locator(y_major_locator)

    plt.tick_params(labelsize=fontsize_d, direction='in')  # 坐标轴字体大小
    plt.tick_params(which='major', length=6, width=2)  # 刻度线

    plt.plot(day, random_d, linewidth = linewidth_d, linestyle='solid', color = 'steelblue', marker = 'v',
             markersize = markersize_, markeredgewidth = markeredge_, label = 'Random', clip_on = False)
    plt.plot(day, topk_d, linewidth=linewidth_d, linestyle='solid', color='purple', marker='^',
             markersize= markersize_, label='HAF', clip_on = False)
    plt.plot(day, ga_d, linewidth=linewidth_d, linestyle='solid', color='green', marker='D',
             markersize= markersize_-2, markeredgewidth=markeredge_, label='GASP', clip_on = False)
    plt.plot(day, kmeans_d, linewidth = linewidth_d, linestyle='solid', color = 'darkorange', marker = 's',
             markersize = markersize_-2, label = 'KMSP', clip_on = False)
    plt.plot(day, ddpg_d, linewidth = linewidth_d, linestyle='solid', color = 'firebrick', marker = 'o',
             markersize = markersize_-1, label = 'DRLO', clip_on = False)

    leg = plt.legend(bbox_to_anchor=(0.03, 0.81), loc='lower left', fontsize=legendsize_d, ncol=5,
               handlelength = 1.5, handletextpad=0.3, borderpad=0.3,
               labelspacing=0.2, columnspacing=0.6, edgecolor='black', framealpha = 1)
    leg.get_frame().set_linewidth(2)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)  # 设置绘图区和画布的边距

    plt.tight_layout()
    img_name = 'test_img/' + time + '/dynamic_img/delay_' + time_month + '_' + str(bs_num) + 'bs_' + str(es_ratio) +'r.png'
    plt.savefig(img_name)
    print("访问时延图绘制完成")

    '''
    工作负载图 - 柱状图
    '''
    # 图像参数设置
    fontsize_w = 24.5
    legendsize_w = 24.5
    bar_width = 0.7

    plt.cla()  # 清除
    plt.figure(figsize=(12, 6))  # 调整图片尺寸
    ax = plt.gca()

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.ylabel("Workload Balancing (min)", fontsize=fontsize_w, labelpad=5)
    ax.yaxis.get_offset_text().set(size=fontsize_w)     #调整左上角数量级字体大小

    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['left'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['top'].set_linewidth(2)  # 设置坐标轴粗细
    ax.spines['right'].set_linewidth(2)  # 设置坐标轴粗细

    plt.tick_params(labelsize=fontsize_w, direction='in')  # 坐标轴字体大小
    plt.tick_params(which='major', length=6, width=2)  # 刻度线


    plt.bar('Random', aver_random_w, width=bar_width, label = 'Random', hatch = '\\', color = 'steelblue', edgecolor = 'k')
    plt.bar('HAF', aver_topk_w, width=bar_width, label='HAF', hatch='+', color='purple', edgecolor='k')
    plt.bar('GSP', aver_ga_w, width=bar_width, label='GASP', hatch='xx', color='green', edgecolor='k')
    plt.bar('KMSP', aver_kmeans_w, width=bar_width, label = 'KMSP', hatch = '//', color = 'darkorange', edgecolor = 'k')
    plt.bar('DRLO', aver_ddpg_w, width=bar_width, label = 'DRLO', hatch = 'xxx', color = 'firebrick', edgecolor = 'k')

    ax.axes.xaxis.set_ticklabels([])    #隐藏刻度标签

    leg = plt.legend(bbox_to_anchor=(0.975, 1.01), loc=1, fontsize=legendsize_w, ncol = 5,
               handlelength = 1.7, handleheight = 0.85, handletextpad=0.2, borderpad=0.3,
               labelspacing=0.4, columnspacing=0.6, edgecolor='black', framealpha = 1)  # 图例
    leg.get_frame().set_linewidth(2)
    plt.ylim((lim_list[2], lim_list[3]))
    plt.subplots_adjust(left=0.34, bottom=0, right=1, top=1)  # 设置绘图区和画布的边距
    plt.tight_layout()
    img_name = 'test_img/' + time + '/dynamic_img/workload_bias_' + time_month + '_' + str(bs_num) + 'bs_' + str(es_ratio) +'r_bar_chart.png'
    plt.savefig(img_name)
    print("工作负载图绘制完成")

