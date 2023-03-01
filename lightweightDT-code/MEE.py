"""
本模块是实现MEE算法的代码

"""

from numpy import *
import numpy as np
from sympy import *
from sympy.abc import x
from scipy import optimize
from scipy.optimize import minimize_scalar
import config
import time

accGlobal = config.accGlobal  # 全局精度
userNum = config.userNum  # 用户数
# 精度向量 初始化为全局精度
accTheta = config.accTheta

fnMax = config.fnMax
fnOpt = config.fnOpt  # 资源决策向量 初始化最大值

Rfn = config.Rfn
Raccn = config.Raccn

cnTask = config.cnTask
parameter = config.parameter  # 电容参数

timeMax = config.timeMax  # 最大时间限制

comE = config.comE  # 通信能耗

comT = config.comT
logweight = config.logweight


# 求qn
def find_qn(acc, fn, n):
    # 分子
    num = (parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logweight * np.log2(100 / (100 - acc)) + \
          comE[
              n, 0] + p * comT[n, 0]
    # 分母
    div = 2 * acc - accGlobal
    return (num / div)


def find_fnOpt(acc, n):
    # 函数fn定义
    fn = lambdify(x, (
            logweight * log(100 / (100 - acc), 2) * (cnTask[n, 0] * parameter * (x ** 2) + p * cnTask[n, 0] / x) +
            comE[n, 0] + p * comT[n, 0]) / (2 * acc - accGlobal))
    # fn下界
    fnLower = (logweight * log2(100 / (100 - acc)) * cnTask[n, 0]) / timeMax

    # fn上界
    fnUpper = np.float64(1 * fnMax[n, 0])

    # 在边界内求最小值
    res = minimize_scalar(fn, bounds=(fnLower, fnUpper), method='bounded')

    return res.x


def find_accOpt(fn, n):
    qn = 6
    count = 0
    # accCur = 0
    while (count < 100):
        count = count + 1
        # 函数acc定义
        accFun = lambdify(x, (
                (parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logweight * log(100 / (100 - x), 2) +
                comE[n, 0] + p * comT[n, 0]) - qn * (2 * x - accGlobal))

        # acc下界
        accLower = accGlobal
        # acc上界
        accUpper = 100 - (100 / (2 ** ((timeMax * fn) / (logweight * cnTask[n, 0]))))

        # 在边界内求最小值
        res = minimize_scalar(accFun, bounds=(accLower, accUpper), method='bounded')
        accCur = res.x
        # 更新qn

        # 计算残差
        valueGn = find_Gn(accCur, fn, qn, n)
        qn = find_qn(accCur, fn, n)

        if abs(valueGn) < 0.0000001:
            break

    return accCur


def find_Gn(acc, fn, qn, n):
    funGn = ((parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logweight * np.log2(100 / (100 - acc)) +
             comE[n, 0] + p * comT[n, 0]) - qn * (2 * acc - accGlobal)

    return funGn


def targetFun(acc, fn, n):
    objValue = (((parameter * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logweight * np.log2(
        100 / (100 - acc)) +
                 comE[n, 0] + p * comT[n, 0])) / (2 * acc - accGlobal)
    return objValue


def energyCost(acc, fn, n):
    Ecost = ((0.001 * cnTask[n, 0] * (fn ** 2) + p * cnTask[n, 0] / fn) * logweight * np.log2(100 / (100 - acc)) + comE[
        n, 0] + p * comT[n, 0])

    return Ecost


def energyEff(acc, fn):
    energyEff = 0
    for n in range(userNum):
        energyEff = energyEff + (2 * acc[n, 0] - accGlobal) / (logweight * log(100 / (100 - acc[n, 0]), 2) * (
                cnTask[n, 0] * parameter * (fn[n, 0] ** 2) + p * cnTask[n, 0] / fn[n, 0]) + comE[n, 0] + p * comT[
                                                                   n, 0])
    return energyEff


def timeCost(acc, fn, n):
    Tcost = ((cnTask[n, 0] / fn) * logweight * np.log2(100 / (100 - acc)))

    return Tcost


if __name__ == '__main__':

    acc_all_p = []
    fn_all_p = []
    for p_ in config.p_enum:
        p = p_
        acc_ = []
        fn_ = []
        energylist = []
        time_cost_max = 0
        global_acc_p = []
        for n in range(userNum):
            time_start = time.time()
            fnOld = np.float64(fnOpt[n, 0])
            # print("初始值fn：",fnOld)
            accOld = 61
            # print("初始值acc：",accOld)
            objDelta = 1
            count = 0
            while (count < 20):
                count = count + 1

                fnNew = find_fnOpt(accOld, n)

                accNew = find_accOpt(fnNew, n)

                objDelta = targetFun(accOld, fnOld, n) - targetFun(accNew, fnNew, n)

                accOld = accNew
                fnOld = fnNew

                totalEnergyCost = energyCost(accNew, fnNew, n)

                energylist.append(totalEnergyCost)
                if abs(objDelta) < 0.000001:
                    break
            time_cost = time.time() - time_start
            if time_cost > time_cost_max:
                time_cost_max = time_cost
            print(str(accNew) + ",")
            acc_.append(accNew)
            fn_.append(fnNew)
        global_acc_p.append(sum(acc_) / len(acc_))
        acc_all_p.append(acc_)
        fn_all_p.append(fn_)

    print("Time Cost: ", time_cost_max)

    file_acc = open("./P2_Acc.pickle", "wb+")
    file_fn = open("./P2_fn.pickle", "wb+")
    acc_all_p = np.array(acc_all_p).reshape((-1, userNum))
    fn_all_p = np.array(fn_all_p).reshape((-1, userNum))
    acc_all_p.dump(file_acc)
    fn_all_p.dump(file_fn)
    file_acc.close()
    file_fn.close()
    print("------------------------------------------")
    print(acc_all_p.tolist())
    print(fn_all_p.tolist())

    print('timeCost', timeCost(76, 32, 1))
