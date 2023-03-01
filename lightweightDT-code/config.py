"""
配置文件模块：
里面包含了算法参数的设置，其他模块引入通过引入该模块实现参数的的导入

"""

import numpy as np

np.random.seed(1)


accGlobal = 60  # 全局精度
userNum = 4  # 用户数
accMax = np.random.rand(userNum, 1) * 10 + 90  # 精度上限

accTheta = np.zeros((userNum, 1))  # 精度向量 初始化为全局精度 ，精度使用百分制表示
accTheta = accTheta + accGlobal

p_enum = range(1, 10, 2)  # delay-sensitive parameter


"""
cnTask、comE、comT 是通过随机的方式生成的
cnTask: 训练一个样本需要的cpu计算量
comE: 通讯能耗
comT： 通讯时间
"""
cnTask = np.array([[3.3], [3.1], [3.0], [3.6], [3.5],
                   [3.2], [3.6], [3.8], [3.0], [2.8],
                   [3.6], [3.1], [2.7], [3.8], [4.0],
                   [3.4], [3.3], [3.1], [3.7], [3.5],
                   [3.5], [2.9], [3.8], [3.6], [2.5],
                   [3.3], [3.0], [3.1], [3.9], [4.0]]) * 10

comE = np.array([[22.93511782], [20.8467719], [22.72520423], [22.91121454], [22.91121454],
                 [21.234235423], [22.2345234], [21.42342442], [22.65756566], [21.965777754],
                 [23.003543545], [21.8467719], [20.4353453453], [22.232323232], [22.5675666],
                 [21.456411782], [22.8467719], [21.92520423], [22.453451454], [21.88712145],
                 [22.434354345], [21.6767674], [22.35453545], [20.345345435], [20.45345453],
                 [20.354534534], [20.435345345], [21.23423423], [21.2345344], [22.078899777],
                 ]) - 15

comT = np.array([[60.79353495], [60.07763347], [60.83838903], [60.370439], [60.370439],
                 [60.454353453], [60.98977777], [60.23432434], [60.598890], [60.234244],
                 [60.893454993], [60.56755333], [60.65754544], [60.0987875], [60.1231245],
                 [60.657456466], [60.87564564], [60.25454557], [60.2349877], [60.7786334],
                 [60.769849553], [60.09349884], [60.34958864], [60.8783499], [60.7874344],
                 [60.089967666], [60.2343566], [60.78978654], [60.34534533], [60.895464933],
                 ]) - 55

weight = 150   # 全局模型收益的权重
logweight = 4  # 局部轮数下界

parameter = 0.001  # 电容参数
timeMax = 4  # 最大时间限制

fnOpt = np.zeros((userNum, 1)) + 6 # 资源决策向量 初始化最大值
Rfn = np.zeros((userNum, 1))  # 对算力的初始化奖励值
Raccn = np.zeros((userNum, 1))  # 对精度的初始化奖励值

eta = 98


outer_max_limit = 100   # LDDIM 最大迭代的代数

# MLE中独有参数

globalCount = 0
modelSize = 10    # 模型大小
step = 0.01       # 步长

totalNewG_limit = 0.000001  # MLE 迭代停止条件

max_round = 20  # MLE 循环轮数

fnMax = np.array([[33], [34], [36], [39], [30], [34]] * 8) + 15  # 最大算力限制