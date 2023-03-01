from utils.batchUtils import *

"""
这里什么没有利用线程来仿真Worker,原因如下：

1： 因为大锁，python线程多并发性能低下
2. 利用进程，更真实仿真了worker


问题：
 1. 本地训练的次数，开始上传模型的条件，是看模型收敛了还是迭代一定次数终止
 2. 每个worker上的数据集怎么分布，从网上下载的训练集有5组，每组10000张，每组中每个类约有1000张，如何利用这些数据，没人应该分多少数据，如何分
 3. 按论文模型的参数设置，模型太大，跑不起来
 4. 使用论文中sgd优化器效果很差，改用Adam，效果有改善，但准确率停在0.5左右
 5. 模型的参数初始化方式
 
"""



# 对文件夹进行清理工作
dirs = [("./work/", "txt"), ("./result/asyn/", "npy")]
for i in dirs:
    clearDir(*i)

maliciousNodes = [1]

if __name__ == "__main__":
    startAllNodesOnFileOnWindows(maliciousNodes)
    # startAllNodesOnFile(maliciousNodes)
    # startAllNodesOnTerminal(maliciousNodes)
