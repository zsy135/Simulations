

# ifconfig

ServerListenIP = "127.0.0.1"  # 要改
ServerListenPort = 50001

ClientIP = "127.0.0.1"
ClientPort = 50001

ClientNum = 2


H_Max = 1000  # 最大本地训练次数
H_Min = 100  # 最小本地训练次数
H_i = [1, 2, 3, 5, 2, 3, 5, 9, 2, 6,8,1, 2, 3, 5, 2, 3, 5, 9, 2, 6,8,1, 2, 3, 5, 2, 3, 5, 9, 2, 6,8,1, 2, 3, 5, 2, 3, 5, 9, 2, 6,8]  # 每个i元素对应设备i的本地训练次数


Alpha = 0.8
ComputeAlphaFun = 0 # 0: const  1: Polynomial   2 Hinge

SheduleFrequency = 4


LearnRate = 0.0001
Rou = 0.005

SavaFreq = 50  # 模型保存频率

EndTime = 10 # 终止时间

BatchSize = 100


TrainDatasetSizePerClient = 500    # 保证能被10整除
TestDatasetSize = 5000

Fed_Asyn = True  # 是否进行异步学习
# Fed_Asyn = False  # 是否进行异步学习
