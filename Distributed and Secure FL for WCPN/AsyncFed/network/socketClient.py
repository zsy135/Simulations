import socket
import time

asServer = False
if asServer:

    serverSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM,0)
    serverSocket.bind(("127.0.0.1",60000))
    serverSocket.listen(1)
    print("Server Start: ")
    client,_ = serverSocket.accept()
    serverSocket.close()

    dataLen = 0
    start = 0

    while True:
        tmp = client.recv(10240)
        dataLen +=tmp
        if start ==0 :
            start = time.time()
        if time.time() - start > 10:
            break
    print("data len: ",dataLen,"rate: ",dataLen/10)
    client.close()
else:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    client.connect(("127.0.0.1", 60000))
    data = 'dfsafsfsfs'.encode()*1024
    start = 0
    while True:
        tmp = client.sendall(data)

