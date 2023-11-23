import csv
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import warnings
import time
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import random
import re
from numpy import *
import math
import community
import jieba
from community import community_louvain
import json
import os



# 预备函数
def jaccard(p, q):
    c=0
    p=set(p)
    q=set(q)
    for i in p:
        if i in q:
            c+=1
    return float(c/(len(p)+len(q)-c))
def isChinese(string):
    for chart in string:
        if chart < u'\u4e00' or chart > u'\u9fff':
            return False
    return True

def checkWord(l):
    res=[]
    for s in l:
        if isChinese(s) == True:
            res.append(s)
    return res
def delete_null(data, columns):
    """
    删除缺失值
    :param data:
    :param columns:
    :return:
    """
    for c in columns:
        data = data[data[c].notna()]
    return data




def preprocess(data):
    """
    数据预处理
    :param data: 是一个包含字典的列表
    """
    data = pd.DataFrame(data)
    # 删除缺失值
    null_column = ['object']  # object是IP
    # 定义告警变量
    alarm_name = 'name'
    # 定义告警发生时间
    alarm_time = 'alarm_time'
    data = data[data['source_name'].isin(['蓝鲸监控平台', '统一监控中心'])]
    data = delete_null(data, null_column)
    data.loc[:, '时间戳'] = data[alarm_time].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
    data = data.sort_values(by='时间戳', ascending=True)
    data = data.reset_index()
    data = data.drop(columns=['index'])
    data.loc[:, '告警编码'] = list(data.loc[:, alarm_name].values)
    # 切分训练级、预测集
    lens = data.shape[0]
    train_rate = 0.6
    train_num = int(lens * train_rate)
    train_data = data.loc[0:train_num]
    test_data = data.loc[train_num + 1:]

    # 对训练集的  告警名称  进行编码
    alarm_count_train = train_data[alarm_name].value_counts()
    variable = ["A{}".format(i+1) for i in range(len(alarm_count_train))]
    alarm_count_train_dict = dict(zip(alarm_count_train.index, variable))
    event_train_dict = {'告警事件': alarm_count_train.index, '发生次数': alarm_count_train.values, '变量名': variable}
    event_train = pd.DataFrame(event_train_dict)
    for i in range(len(event_train)):
        train_data["告警编码"].replace(event_train["告警事件"][i], event_train["变量名"][i], inplace=True)

    # 处理测试数据中，不在训练告警类型中,采用A0进行编码
    test_data = test_data.sort_values(by='时间戳', ascending=True)
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])
    f = lambda x: "A0" if x not in alarm_count_train_dict.keys() else alarm_count_train_dict[x]
    test_data['告警编码'] = test_data['告警编码'].apply(f)
    print('data processing end')
    return train_data, test_data, event_train



def segment_alarm(train_data, event_train, segword_path, jaccard_value=0.6, time_value=60):
    """
    告警分割
    :param train_data: 训练集数据（preprocess函数返回）
    :param event_train: 预处理后的数据（preprocess函数返回）
    :param segword_path: 用户自定义词典文件路径
    """
    alarm_name = 'name'
    jieba.load_userdict(segword_path)
    count = 0
    train_data['分段'] = ''
    train_data['分段'][0] = 1
    segment = []
    seg = [train_data['告警编码'][0]]
    for i in range(1, len(train_data)):
        time1 = train_data['时间戳'][i - 1]
        time2 = train_data['时间戳'][i]
        room1 = train_data[alarm_name][i - 1]
        room2 = train_data[alarm_name][i]
        room1_word = list(jieba.cut(room1, cut_all=False))
        room2_word = list(jieba.cut(room2, cut_all=False))
        room_jaccard = jaccard(room1_word, room2_word)
        time_interval = time2 - time1
        if room_jaccard >= jaccard_value and time_interval <= time_value:
            train_data['分段'][i] = train_data['分段'][i - 1]
        else:
            train_data['分段'][i] = train_data['分段'][i - 1] + 1
    print("数据分段结束")

    # 方向矩阵
    direct = mat(zeros((len(event_train), len(event_train))))
    joint_count = mat(zeros((len(event_train), len(event_train))))
    group_id = 1
    count = 0
    while count < len(train_data):
        g = []
        gmap = {}
        gid = train_data['分段'][count]
        for i, e in zip(train_data['分段'][count:], train_data['告警编码'][count:]):
            if i == gid:
                g.append(e)
                if e not in gmap:
                    gmap[e] = 1
                else:
                    gmap[e] += 1
            else:
                break
        count = count + len(g)
        # 计算方向矩阵
        for x in range(len(g)):
            i = int(re.findall("\d+", g[x])[0]) - 1
            for y in range(x + 1, len(g)):
                j = int(re.findall("\d+", g[y])[0]) - 1
                if i != j:
                    direct[i, j] += 1
        # 计算共现矩阵
        for k, v in gmap.items():
            i = int(re.findall("\d+", k)[0]) - 1
            for kk, vv in gmap.items():
                if k != kk:
                    j = int(re.findall("\d+", kk)[0]) - 1
                    joint_count[i, j] = joint_count[i, j] + v
    print("方向矩阵、共现矩阵计算结束")

    # 计算条件概率
    arr = []
    for i in range(len(event_train)):
        arr.append('A' + str(i + 1))
    event_cond_p = pd.DataFrame(index=arr, columns=arr)
    for i in event_cond_p.index:
        for j in event_cond_p.columns:
            x = int(re.findall("\d+", i)[0])
            y = int(re.findall("\d+", j)[0])
            if joint_count[x - 1, y - 1] == 0:
                event_cond_p[i][j] = 0
            else:
                if int(event_train[(event_train['变量名'] == i)]['发生次数'].values[0]) > 3 and int(
                        event_train[(event_train['变量名'] == j)]['发生次数'].values[0]) > 3:
                    p = max(
                        joint_count[x - 1, y - 1] / int(event_train[(event_train['变量名'] == i)]['发生次数'].values),
                        joint_count[y - 1, x - 1] / int(event_train[(event_train['变量名'] == j)]['发生次数'].values))
                else:
                    p = (joint_count[x - 1, y - 1] + joint_count[y - 1, x - 1]) / (
                            int(event_train[(event_train['变量名'] == i)]['发生次数'].values) + int(
                        event_train[(event_train['变量名'] == j)]['发生次数'].values))
                if direct[x - 1, y - 1] >= direct[y - 1, x - 1]:
                    event_cond_p[i][j] = p
                    event_cond_p[j][i] = 0
                else:
                    event_cond_p[i][j] = 0
                    event_cond_p[j][i] = p

    correlation = event_cond_p
    return correlation



class PRIterator:
    __doc__ = '''计算一张图中的PR值'''

    def __init__(self, dg, sim):
        self.damping_factor = 0.85  # 阻尼系数,即α
        self.max_iterations = 100  # 最大迭代次数
        self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
        self.graph = dg
        self.sim = sim

    def page_rank(self):
        #  先将图中没有出链的节点改为对所有节点都有出链
        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
        damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分

        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                neighbers = []
                for i in nodes:
                    if self.graph.has_edge(node, i):
                        neighbers.append(i)
                for outcident_page in neighbers:  # 遍历所有“入射”的页面
                    if self.graph.in_degree(outcident_page) == 0:
                        rank += 0
                        print(outcident_page)
                    else:
                        rank += self.damping_factor * (
                                    page_rank[outcident_page] / self.graph.in_degree(outcident_page)) * self.sim[node][
                                    outcident_page]
                neighbers2 = []
                for j in nodes:
                    if self.graph.has_edge(j, node):
                        neighbers2.append(j)
                for outcident_page in neighbers2:
                    if self.graph.in_degree(node) == 0:
                        rank -= 0
                    else:
                        rank -= self.damping_factor * (page_rank[node] / self.graph.in_degree(node)) * \
                                self.sim[outcident_page][node]

                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank

            if change < self.min_delta:
                break
        sort_pr = sorted(page_rank.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return sort_pr



def root_cause(event_cond_p, event_train):
    """
    根因定位实现
    :param event_cond_p: 关联数据（segment_alarm函数返回）
    :param event_train: 预处理后的数据（preprocess函数返回）
    :return result_rc: 根因定位的结果json
    :return train_result: 有向图的相关统计信息
    :return each_community_node,G,colors: 用于绘制社区图的相关信息
    """
    sim = event_cond_p
    # 设置有向图
    G = nx.DiGraph()
    # 设定消除边的阈值
    theta = 0.01
    labels = {}
    # 根据画边
    G.add_nodes_from(sim.index)
    for i in sim.index:
        ii = int(re.findall("\d+", i)[0])
        for j in sim.columns:
            jj = int(re.findall("\d+", j)[0])
            if sim[i][j] > theta:
                G.add_weighted_edges_from([(i, j, sim[i][j])])
                label_key = (i, j)
                labels[(i, j)] = sim[i][j]

    # 转化为无向图
    G1 = G.to_undirected()

    # 消除独立的节点并标记
    node_visit = [0] * len(event_train)
    # 消除没有邻居的节点
    ind_node = []
    for i in list(G.nodes):
        if G.in_degree(i) == 0 and G.out_degree(i) == 0:
            x = int(re.findall("\d+", i)[0])
            ind_node.append(i)
            G.remove_node(i)
            node_visit[x - 1] = 1

    # 统计图的情况
    node_num = len(G.nodes)  # 节点数
    edge_num = len(G.edges)  # 连接边数

    # 划分子图并存储
    node_outnum = {}
    for node in G.nodes:
        node_outnum[node] = G.out_degree(node)

    # 按每个节点的出度进行排序
    node_outnum_rank = list(sorted(node_outnum.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    hash_map = {}
    event_variable = {}  # 记录键值对
    for i in range(len(event_train)):
        event_variable[event_train.loc[i, '变量名']] = event_train['告警事件'][i]
        hash_map[i] = []
    sub_num = 1
    label = {}
    count = 1
    a = {}
    alarm_variable = []
    alarm_name = []
    community = []
    root_cause = []

    for node, nodenum in node_outnum_rank:
        # 没有被访问过
        subarrs = sorted(list(nx.bfs_tree(G1, node)))  # nx.bfs_tree(G1, node)以广度优先获取有向图的节点列表
        sub2 = G1.subgraph(subarrs)
        # 执行louvain算法
        partition = community_louvain.best_partition(sub2, weight='weight', random_state=9000)
        node_label = partition[node]
        subarr = []
        # 得到和当前节点社区一致的节点
        for la in partition.keys():
            lai = int(re.findall("\d+", la)[0]) - 1
            if partition[la] == node_label:
                subarr.append(la)

        # 重新生成子图
        sub = G.subgraph(subarr)
        # 根据相应的情况得到了子图的节点列表 标记了属于哪个图
        # 标记根因
        pr = PRIterator(sub, sim)
        rcnode = pr.page_rank()

        # 将根因pr得分转化成百分比
        sum = 0
        for r in rcnode:
            sum += r[1]
        rc = []
        for r in rcnode:
            rc.append((r[0], "{:.4}%".format(r[1] / sum * 100)))
        a[node] = [sub.nodes(), rc]

        alarm_variable.append(node)
        alarm_name.append(event_variable[node])
        community.append(list(sub.nodes()))
        root_cause.append(rc)
    result = {'alarm': alarm_variable,
              'alarm_name': alarm_name,
              'community': community,
              'root_cause': root_cause}
    # 有向图相关的统计数据
    train_result = pd.DataFrame(result)

    # result_rc离线学习的结果,colors绘制社区图的颜色定义,each_community_node每个社区包含的节点
    colors = []
    each_community_node = []
    result_rc = {}
    c = 1
    for k, v in a.items():
        color = []
        if v[0] not in each_community_node:
            keys = "community_{}".format(c)
            each_community_node.append(v[0])
            alarm_rc = {"alarm_var": list(v[0]), "root_cause": v[1]}
            result_rc[keys] = alarm_rc
            print("{}社区：{}, 根因：{}".format(c, v[0], v[1]))
            color = ["blue"] * len(v[0])
            color[list(v[0]).index(v[1][0][0])] = 'red'
            colors.append(color)
            c += 1

    print(result_rc)
    # 返回根因分析结果
    return result_rc, train_result, each_community_node, G, colors




def showroot(path, each_community_node, G, colors):
    """
    对每个社区绘制拓扑图，并将根因标红
    :param path: 图保存的路径
    :param each_community_node: 社区节点信息（root_cause函数返回）
    :param G: 图信息（root_cause函数返回）
    :param colors: colors信息（root_cause函数返回）
    """
    for i, nodes in enumerate(each_community_node):
        louvain_sub = G.subgraph(nodes)
        nx.draw_networkx(louvain_sub, node_color=colors[i])
        filename = '社区_{}.png'.format(i + 1)
        filepath = os.path.join(path, filename)
        plt.savefig(filepath)
        plt.close()




def root_cause_analysis(data, segword_path):
    """
    完整的根因定位
    :param data: 输入的元素为字典的list对象
    :param segword_path: 用户自定义词典文件路径
    :return: 根因定位所有相关的结果
    """
    train_data = []
    test_data = []
    # 数据集分割、预处理
    train_data, test_data, event_train = preprocess(data)
    # 获取关联关系
    correlation = segment_alarm(train_data, event_train, segword_path)
    # 根因分析
    result, train_result, each_community_node, G, colors = root_cause(correlation, event_train)
    # 返回结果json
    return result, train_result, each_community_node, G, colors