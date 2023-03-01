import time

import psutil
import threading
import tensorflow as tf
import os
sess = tf.Session()
w = tf.ones((10000, 10000))
mul_w = tf.matmul(w, w)


sess.run(tf.global_variables_initializer())

stop_print_fre = False
writer = open("./CPU_Percent.txt", "w+")

p = psutil.Process(os.getpid())


def print_fre():
    global stop_print_fre
    global p
    while not stop_print_fre:
        writer.write(str(p.cpu_percent(0.2))+"\n")
    writer.close()


t = threading.Thread(target=print_fre)
t.start()
start = time.time()
sess.run(mul_w)
stop_print_fre = True
print(time.time() - start)
