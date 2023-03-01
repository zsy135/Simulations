import hashlib
import time


found = False

currentVsersion = 10
newestVersion = 20
nonce = 0
timeCost = []
for i in range(256,256-20,-1):
    start = time.time()
    target = 2 ** i
    while True:
        hash_result = hashlib.sha256(
            (str(currentVsersion) + str(newestVersion) + str(nonce)).encode("utf-8")).hexdigest()
        if int(hash_result, 16) < target:
            break
        nonce +=1
    tmp = time.time() - start
    timeCost.append(tmp)
    print(tmp)
import matplotlib.pyplot as plt
plt.plot(timeCost)
plt.show()
print("Cost: ",timeCost)





