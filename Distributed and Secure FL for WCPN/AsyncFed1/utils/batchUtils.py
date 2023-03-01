import time
import config
import subprocess
import os


def clearDir(path, endStr=""):
    if not path.endswith("/"):
        path += "/"
    todel = os.listdir(path)
    for file in todel:
        if file.endswith(endStr):
            os.remove(path + file)


def getWriters(num, nameTmpl):
    tmp = []
    for i in range(num):
        name = nameTmpl.format(i)
        writer = open(name, "w+")
        tmp.append(writer)
    return tmp


def closeWriters(writers):
    for i in writers:
        if i and not i.closed:
            i.close()
    return True


# 这种方式只适合在Ubuntu上运行
def startAllNodesOnTerminal(maliciousNodes):
    nodesProc = []
    for i in range(config.ClientNum):
        if i in maliciousNodes:
            proc = subprocess.Popen(
                ["gnome-terminal", "--", "/bin/bash", "-c", "python3.7", "startWorker.py", str(i), 'true'])
        else:
            proc = subprocess.Popen(["gnome-terminal", "--", "/bin/bash", "-c", "python3.7", "startWorker.py", str(i)])
        nodesProc.append(proc)
        print("------- Start Node{} -------".format(i))
        time.sleep(0.5)

    time.sleep(10)
    while True:
        stoped = input("if stop all nodes:(y) ").strip().lower()
        if stoped == "y":
            break
        else:
            continue
            # 关闭所有节点
    for p in nodesProc:
        p.kill()


def startAllNodesOnFile(maliciousNodes):
    outFileNameTmpl = "./work/Node{}_OutPut.txt"
    writers = getWriters(config.ClientNum, outFileNameTmpl)
    nodesProc = []

    # 以文件作为每个节点的输出
    # 开启节点
    for i in range(config.ClientNum):
        if i in maliciousNodes:
            proc = subprocess.Popen(["python3.7", "startWorker.py", str(i), 'true'], stdout=writers[i])
        else:
            proc = subprocess.Popen(["python3.7", "startWorker.py", str(i)], stdout=writers[i])
        nodesProc.append(proc)
        print("------- Start Node{} -------".format(i))
        time.sleep(0.5)

    time.sleep(10)

    while True:
        stoped = input("if stop all nodes:(y) ").strip().lower()
        if stoped == "y":
            break
        else:
            continue
    # 关闭所有节点
    for p in nodesProc:
        p.kill()

    closeWriters(writers)


def startAllNodesOnFileOnWindows(maliciousNodes: object) -> object:
    outFileNameTmpl = "./work/Node{}_OutPut.txt"
    writers = getWriters(config.ClientNum, outFileNameTmpl)
    nodesProc = []

    # 以文件作为每个节点的输出
    # 开启节点
    for i in range(config.ClientNum):
        if i in maliciousNodes:
            proc = subprocess.Popen(["python", "startWorker.py", str(i), 'true'], stdout=writers[i])
        else:
            proc = subprocess.Popen(["python", "startWorker.py", str(i)], stdout=writers[i])
        nodesProc.append(proc)
        print("------- Start Node{} -------".format(i))
        time.sleep(0.5)

    time.sleep(10)
    while True:
        stoped = input("if stop all nodes:(y) ").strip().lower()
        if stoped == "y":
            break
        else:
            continue
    # 关闭所有节点
    for p in nodesProc:
        p.kill()

    closeWriters(writers)
