import socket
import time


class Connect:
    def __init__(self, ip=None, port=None, socket_=None, rIp=None, rPort=None):
        self.Protocal = "TCP"
        self.IP = ip
        self.port = port
        self.remoteIP = rIp
        self.remotePort = rPort
        if socket_ is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            # self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
            self.binded = False
            self.connected = False

        else:
            self.socket = socket_
            self.binded = True
            self.connected = True


    def close(self):
        self.socket.close()

    def getPeerName(self):
        if self.remoteIP is None:
            self.remoteIP, self.remotePort = self.socket.getpeername()
        else:
            return self.remoteIP, self.remotePort


    def bind(self):
        self.binded = True
        self.socket.bind((self.IP, self.port))


    def connect(self, rIP=None, rPort=None):
        if rIP is not None:
            self.remoteIP = rIP
        if rPort is not None:
            self.remotePort = rPort
        times = 0
        while times < 10:
            try:
                 self.socket.connect((self.remoteIP, self.remotePort))
            except Exception:
                time.sleep(0.5)
            else:
                self.connected = True
                break
            times += 1

    def listen(self):
        if self.binded:
            self.socket.listen()
        else:
            print("*** Socket Start Listen Before Bind !!!")

    def accept(self):
        return self.socket.accept()

    def send(self, data):
        if not self.connected:
            print("*** Use unconnected socket send data !!!")
            return
        self.socket.send(len(data).to_bytes(4, "little"))  # 发送即将发送数据的长度
        # while len(data) > 0:
        #     l = self.socket.send(data)  # 发送l字节长的数据
        #     data = data[l:]
        self.socket.sendall(data)
        return len(data)

    def recieve(self):
        if not self.connected:
            print("*** Use unconnected socket recieve data !!!")
            return
        l = self.socket.recv(4)
        l = int.from_bytes(l, "little")  # 接受即将接受数据的长度
        ret = bytes()
        while l>0:
            tmp = self.socket.recv(l)
            l -= len(tmp)
            ret += tmp
        return ret
