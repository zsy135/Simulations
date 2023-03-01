import numpy
from scipy import optimize
from numpy import *
import numpy as np
from sympy import *
from sympy.abc import x
from scipy.optimize import minimize_scalar

import config
import time

accGlobal = config.accGlobal  # 全局精度
userNum = config.userNum  # 用户数
accMax = config.accMax
fnMax = config.fnMax
fnOpt = config.fnMax
accTheta = config.accTheta
cnTask = config.cnTask
comE = config.comE
comT = config.comT
step = config.step
x_list = []
count_list = []
globalCount = config.globalCount
weight = config.weight
logweight = config.logweight


def find_eta(acc):
    num = 0  # 分子
    div = 0  # 分母
    for n in range(userNum):
        num = num + weight * log(1 + (acc[n, 0] - accGlobal), 2)
        div = div + (cnTask[n, 0] / fnOpt[n, 0]) * logweight * log(100 / (100 - acc[n, 0]), 2) + comT[n, 0]
    return (num / div).evalf()


def solveP1(eta):
    res_list = []
    for n in range(userNum):
        # 函数fn定义
        fn = lambdify(x, -weight * (log(1 + (x - accGlobal), 2)) + eta * (
                    (cnTask[n, 0] / fnOpt[n, 0]) * logweight * log(100 / (100 - x), 2) + 5))
        res = minimize_scalar(fn, bounds=(accGlobal, 99.999), method='bounded')

        res_list.append(res.x)
    return res_list


def find_G(acc, eta):
    funG = 0
    for n in range(userNum):
        funG = funG + (- weight * log(1 + (acc[n, 0] - accGlobal), 2) + eta * (
                    (cnTask[n, 0] / fnOpt[n, 0]) * logweight * log(100 / (100 - acc[n, 0]), 2) + comT[n, 0]))
    return funG.evalf()


if __name__ == '__main__':
    acc_all_round = []
    acc_global_round = []

    for _ in range(config.round):

        eta = config.eta
        accTheta = None
        time_start = time.time()
        for i in range(config.max_round):
            a = solveP1(eta)


            accTheta = np.array(a).reshape((-1, 1))
            print("acc:\n", accTheta)
            # 更新eta

            # 计算残差
            totalNewG = find_G(accTheta, eta)


            eta = find_eta(accTheta)

            print("eta:", eta)
            print("total:", totalNewG)
            # print("funcValue",funcValue)
            if abs(totalNewG) < config.totalNewG_limit:
                break
        time_cost = time.time() - time_start
        accTheta = np.array(accTheta).reshape((-1,)).tolist()
        print(accTheta)
        print("Time Cost: ", time_cost)
        acc_all_round.append(accTheta)
        accGlobal = sum(accTheta) / (len(accTheta))
        acc_global_round.append(accGlobal)
    print(acc_all_round)
    print(acc_global_round)
