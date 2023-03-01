from worker import Worker
import sys

# 利用该方式启动是为了方便调试，每个worker 都有一个调试界面
# 程序调试完成后，利用workerMain.py 启动worker 更加方便


if __name__ == "__main__":
    myid = 0
    asMalicious = False
    if len(sys.argv) == 2:
        myid = int(sys.argv[1])
    if len(sys.argv) == 3:
        tmp = sys.argv[2].lower().strip()
        if tmp == "true":
            asMalicious = True
    print("Node{} Start (Is Good: {})---------------".format(myid,not asMalicious))
    worker = Worker(myid, malicious=asMalicious)
    worker.start()
    # worker = Worker(0)
    # worker.start()
