from server import Server
from utils.batchUtils import clearDir

dirs = [("./result/roundacc", "npy"), ("./result/asyn/", "npy")]
for i in dirs:
    clearDir(*i)


server = Server()
server.start()     # 启动服务端
