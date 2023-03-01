import socket


def get_socket():
    try:
        s_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    except OSError:
        s_ = None
    return s_


def receive_data(s_: socket.socket):
    d_ = s_.recv(10240)
    return d_


def get_data_socket(address_=""):
    s = get_socket()
    # address_ = socket.gethostbyname(socket.gethostname())
    if address_ == "":
        address_ = "127.0.0.1"
    s.bind((address_, 0))
    address_ip, port = s.getsockname()
    return s, address_ip, port
