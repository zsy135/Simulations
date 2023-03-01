import threading
import time
from threading import Lock


class RWLock:
    def __init__(self):
        self._r_mutex = Lock()
        self._w_mutex = Lock()
        self._reader_num = 0

    def reader_acquire(self):
        self._r_mutex.acquire()
        self._reader_num += 1
        if self._reader_num == 1:
            _got = False
            while not _got:
                _got = self._w_mutex.acquire()
        self._r_mutex.release()
        return True

    def reader_release(self):
        self._r_mutex.acquire()
        self._reader_num -= 1
        if self._reader_num == 0:
            self._w_mutex.release()
        self._r_mutex.release()

    def writer_acquire(self):
        return self._w_mutex.acquire()

    def writer_release(self):
        return self._w_mutex.release()


class Condition:
    def __init__(self):
        self._waiters = []
        self._lock = threading.Lock()

    def wait(self):
        self._lock.acquire()
        l_ = threading.Lock()
        l_.acquire()
        self._waiters.append(l_)
        self._lock.release()
        l_.acquire()

    def notify(self):
        self._lock.acquire()
        try:
            l_: threading.Lock = self._waiters.pop()
        except IndexError:
            pass
        else:
            l_.release()
        self._lock.release()

    def notifyAll(self):
        self._lock.acquire()
        while len(self._waiters) != 0:
            l_: threading.Lock = self._waiters.pop()
            l_.release()
        self._lock.release()


# test
if __name__ == "__main__":
    # test RWLock
    # def fun(rw: RWLock):
    #     print("thread start")
    #     rw.writer_acquire()
    #     print("get writer")
    #     time.sleep(2)
    #     rw.writer_release()
    #     print("release writer\n")
    #
    #
    # def fun1(rw: RWLock):
    #     time.sleep(1)
    #     rw.reader_acquire()
    #     print("get reader\n")
    #     rw.reader_release()
    #
    #
    # rw = RWLock()
    # t = threading.Thread(target=fun, args=(rw,))
    # t.start()
    #
    # t1 = threading.Thread(target=fun1, args=(rw,))
    # t1.start()
    #
    # t2 = threading.Thread(target=fun1, args=(rw,))
    # t2.start()
    #
    # t3 = threading.Thread(target=fun1, args=(rw,))
    # t3.start()

    # test Condition

    l = []
    con = Condition()

    def t1(con: Condition, l: list):
        while len(l) == 0:
            con.wait()
        l.append(threading.current_thread().name)
        print(threading.current_thread().name, ": ", l)


    t_list = []
    t = threading.Thread(target=t1, args=(con, l))
    t.start()
    t_list.append(t)
    t = threading.Thread(target=t1, args=(con, l))
    t.start()
    t_list.append(t)
    # t = threading.Thread(target=t1, args=(con, l))
    # t.start()
    # t_list.append(t)

    # s = input("in :")
    l.append("start: ")
    finished = False
    con.notifyAll()
    while not finished:
        #time.sleep(0.5)
        con.notifyAll()
        # con.notify()
        # n = 0
        # for t in t_list:
        #     if not t.is_alive():
        #         n += 1
        # if n == 2:
        #     finished = True
