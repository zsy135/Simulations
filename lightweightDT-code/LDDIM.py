"""
本模块是 LDDIM 算法的代码实现


"""
from numpy import *
import numpy as np
from sympy import *
from sympy.abc import x
from scipy.optimize import minimize_scalar
import time
from functools import wraps
import psutil
import os
import threading
import config

CPU_FREQ = psutil.cpu_freq().max  # GHZ

process = psutil.Process(os.getpid())
wait_record_cpu_percent_cond = threading.Condition()
is_set_recod_stoped = False
event_record_cpu_p_finished = threading.Event()
event_record_cpu_p_finished.clear()
cpu_percent_record_data = []
caiji = 0.1  # CPU 负载采集频率

Energy = []
Time = []

Data_Recoder = []


def get_cpu_percent():
    global is_set_recod_stoped
    while not is_set_recod_stoped:
        with wait_record_cpu_percent_cond:
            wait_record_cpu_percent_cond.wait()
        while event_record_cpu_p_finished.is_set():
            percent_ = (process.cpu_percent(caiji) / 100) * CPU_FREQ
            # CPU_percent_writer.write(str(percent_) + ", ")
            cpu_percent_record_data.append(percent_)


t = threading.Thread(target=get_cpu_percent)
t.start()


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        event_record_cpu_p_finished.set()
        with wait_record_cpu_percent_cond:
            wait_record_cpu_percent_cond.notifyAll()
        time.sleep(0.1)
        result = function(*args, **kwargs)
        t1 = time.time() - 0.1
        event_record_cpu_p_finished.clear()
        cpu_f = 0
        energy = 0
        for i in cpu_percent_record_data:
            energy += 1e-10 * caiji * i ** 3

        cpu_percent_record_data.clear()
        Data_Recoder.append([(t1 - t0), energy])
        print("Total time running %s : %s seconds ; CPU  %s" %
              (function.__name__, str(t1 - t0), energy)
              )
        # print("DATA:   ", Data_Recoder)
        return result

    return function_timer


accGlobal = config.accGlobal  # 全局精度
userNum = config.userNum  # 用户数
accTheta = config.accTheta  # 精度向量 初始化为全局精度

fnMax = config.fnMax
fnOpt = config.fnMax  # 资源决策向量 初始化最大值

Rfn = config.Rfn
Raccn = config.Raccn

parameter = config.parameter

# 能耗与时间权重
timeMax = config.timeMax  # 最大时间限制

cnTask = config.cnTask
comE = config.comE
comT = config.comT
parameter = config.parameter
weight = config.weight
logWeight = config.logweight


def fun_u(acc, fn, n, qn):
    objValue = ((parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logWeight * log(100 / (100 - acc), 2) +
                comE[n, 0] + p * comT[n, 0]) - qn * (2 * acc - accGlobal)
    return objValue.evalf()


def fun_Hn(acc, fn, Rfn, Raccn, n, qn):
    Hn = fun_u(acc, fn, n, qn) - (Rfn * fn) - (Raccn * acc)
    return Hn


# @fn_timer
def find_Gn(acc, eta, fn, n):
    funG = - weight * log(1 + acc - accGlobal, 2) + eta * (
            (cnTask[n, 0] / fn) * logWeight * log(100 / (100 - acc), 2) + comT[n, 0])
    return funG.evalf()


def fun_Phi(fn, acc, eta, Rfn, Raccn, n, qn):
    Phi = find_Gn(acc, eta, fn, n) + fun_Hn(acc, fn, Rfn, Raccn, n, qn)
    return Phi


def find_fnOpt(acc, n, eta, Rfn, Raccn, qn):
    # 函数fn定义
    fn = lambdify(x, fun_Phi(x, acc, eta, Rfn, Raccn, n, qn))
    # fn下界
    fnLower = (logWeight * log2(100 / (100 - acc)) * cnTask[n, 0]) / timeMax
    # fn上界
    fnUpper = np.float64(1 * fnMax[n, 0])
    # 在边界内求最小值
    res = minimize_scalar(fn, bounds=(fnLower, fnUpper), method='bounded')
    # print("fn:",res.x)
    return res.x


def find_accOpt(fn, n, eta, Rfn, Raccn, qn):
    count = 0
    valueOld = 0
    while (count < 50):
        count = count + 1
        # 函数acc定义
        accFun = lambdify(x, fun_Phi(fn, x, eta, Rfn, Raccn, n, qn))

        # acc下界
        accLower = accGlobal
        # acc上界
        accUpper = 100 - (100 / (2 ** ((timeMax * fn) / (logWeight * cnTask[n, 0]))))
        # accLower= np.float64(accLower)
        # print("acc上界：",accUpper)

        # 在边界内求最小值
        res = minimize_scalar(accFun, bounds=(accLower, accUpper), method='bounded')
        accCur = res.x
        # print("accCur:",accCur)

        # 计算残差
        valueNew = fun_Phi(fn, accCur, eta, Rfn, Raccn, n, qn)

        valueDelta = valueNew - valueOld
        valueOld = valueNew
        # 更新qn
        qn = find_qn(accCur, fn, n)

        if abs(valueDelta) < 0.0000001:
            break

    return accCur


def find_qn(acc, fn, n):
    # 分子
    num = (parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logWeight * log(100 / (100 - acc), 2) + comE[
        n, 0] + p * comT[n, 0]
    # 分母
    div = 2 * acc - accGlobal
    return (num / div).evalf()


def find_eta(acc, fn):
    num = 0  # 分子- weight*log(1+acc-accGlobal,2) + eta*((cnTask[n,0]/fnOpt[n,0])*log(100/(100-acc),2)+modelSize/rn)
    div = 0  # 分母
    for n in range(userNum):
        num = num + log(1 + acc[n, 0] - accGlobal, 2)
        div = div + (cnTask[n, 0] / fn[n, 0]) * logWeight * log(100 / (100 - acc[n, 0]), 2) + (comT[n, 0])

    return ((weight * num) / div).evalf()


def find_Rfn(acc, fn, Rfn, Raccn, n, qn):
    # print("aggregator incents Fn", end=' ')
    x = symbols('x')
    # 符号x，自变量
    y = fun_Hn(acc, x, Rfn, Raccn, n, qn)  # 公式

    dify = diff(y, x)  # 求导
    # print(dify) #打印导数

    # 给定x值，求对应导数的值
    Rfn = dify.subs('x', fn)
    return Rfn.evalf()


def find_Raccn(acc, fn, Rfn, Raccn, n, qn):
    # print("aggregator incents Acc", end=' ')
    x = symbols('x')
    # 符号x，自变量
    y = fun_Hn(x, fn, Rfn, Raccn, n, qn)  # 公式

    dify = diff(y, x)  # 求导
    # print(dify)  #打印导数

    # 给定x值，求对应导数的值
    Raccn = dify.subs('x', acc)
    return Raccn.evalf()


def energyCost(acc, fn, n):
    Ecost = ((parameter * cnTask[n, 0] * (fn ** 2)) * logWeight * np.log2(100 / (100 - acc)) + comE[n, 0])
    return Ecost


def timeCost(acc, fn):
    Tcost = ((cnTask / fn) * logWeight * np.log2(100 / (100 - acc)))
    return Tcost


def userCost(acc, fn, accGlobal):
    Ecost = 0
    for n in range(0, 4):
        Ecost = Ecost + (((parameter * cnTask[n, 0] * (fn[n, 0] ** 2)) * logWeight * np.log2(100 / (100 - acc[n, 0])) +
                          comE[n, 0]) + p * (comT[n, 0] + (
                (cnTask[n, 0] / fn[n, 0]) * logWeight * np.log2(100 / (100 - acc[n, 0])))))
    return Ecost / 4


@fn_timer
def clientInter(accOld, n, eta, Rfn, Raccn, qnOld, Phi_Old):
    print("client  %s" % (n), end=' ')
    count = 0
    while (count < 30):
        count = count + 1
        fnNew = find_fnOpt(accOld, n, eta, Rfn, Raccn, qnOld)
        # print("fnNew: ", fnNew)
        accNew = find_accOpt(fnNew, n, eta, Rfn, Raccn, qnOld)
        # print("accNew: ",accNew)
        Phi_New = fun_Phi(fnNew, accNew, eta, Rfn, Raccn, n, qnOld)
        objDelta = Phi_New - Phi_Old
        qnNew = find_qn(accNew, fnNew, n)
        accOld = accNew
        fnOld = fnNew
        qnOld = qnNew
        Phi_Old = Phi_New
        # print("Delta: ",objDelta)
        if (abs(objDelta) < 0.0001):
            break
    return accNew, fnNew, qnNew


@fn_timer
def AggreInter(acc, fn, Rfn, Raccn, qnNew, eta, totalOld):
    for n in range(userNum):
        Rfn[n, 0] = 0.5 * find_Rfn(acc[n, 0], fn[n, 0], Rfn[n, 0], Raccn[n, 0], n, qnNew)
        Raccn[n, 0] = 0.5 * find_Raccn(acc[n, 0], fn[n, 0], Rfn[n, 0], Raccn[n, 0], n, qnNew)
    # 计算残差
    totalNew = 0
    for n in range(userNum):
        totalNew = totalNew + find_Gn(fn[n, 0], acc[n, 0], eta, n)
    totalDelta = abs(totalNew - totalOld)
    totalOld = totalNew
    # 更新eta
    eta = find_eta(acc, fn)
    print("eta", eta)
    print("Rfn", Rfn)
    print("Raccn", Raccn)
    print("totalDelta", totalDelta)

    return eta, Rfn, Raccn, totalDelta, totalOld


"""
本算法实际上是并行计算，但为了不增加程序的复杂度
这里实际上是串行运算
"""
if __name__ == '__main__':

    acc_all_p = []
    fn_all_p = []

    acc = []
    global_acc = []
    fn = []

    # for p in config.p_enum:可以调节不同的delay-sensitive parameter观察对实验的影响
    p = 4  # delay-sensitive parameter
    eta = config.eta
    outer = 0  # 外层循环轮数
    cancha = []
    totalOld = 0
    time_start = time.time()
    #
    acc_p = []
    global_acc_p = []
    fn_p = []

    timeMax = 50  # 时间限制

    fn_all = []  # 记录每个外层循环用户的算力
    acc_all = []  # 记录每个外层循环用户的精度
    Rfn_all = []  # 记录每个外层循环对算力的奖励值
    Racc_all = []  # 记录每个外层循环对精度的奖励值

    while outer < config.outer_max_limit:
        time_start = time.time()

        outer = outer + 1
        print("outer:", outer)
        acc_list = []
        fn_list = []

        # 更新acc和fn （eta未更新）
        for n in range(userNum):
            fnOld = np.float64(fnOpt[n, 0])
            # print("初始值fn：",fnOld)
            accOld = 61
            # print("初始值acc：",accOld)
            qnOld = 5
            Phi_Old = 0
            accNew, fnNew, qnNew = clientInter(accOld, n, eta, Rfn[n, 0], Raccn[n, 0], qnOld, Phi_Old)
            acc_list.append([accNew])
            fn_list.append([fnNew])
        # 整合所有用户acc和fn
        acc = np.array(acc_list)
        fn = np.array(fn_list)
        print("acc", acc)
        print("fn", fn)
        fn_all.append(fn[0, 0])
        acc_all.append(acc[0, 0])

        eta, Rfn, Raccn, totalDelta, totalOld = AggreInter(acc, fn, Rfn, Raccn, qnNew, eta, totalOld)
        Rfn_all.append(Rfn)
        Racc_all.append(Raccn)
        if abs(totalDelta) < 0.01:
            break
        print(time.time() - time_start)
    print("算法收敛结束！")
    print("Reward: ", np.array(Racc_all).tolist(), np.array(Rfn_all).tolist())  # 输出全部循环的激励结果
    print("Accuracy: ", acc_all, fn_all)  # 输出全部循环的精度和算力结果
    is_set_recod_stoped = True

