import pickle
import random
import numpy as np

import config
# from tensorflow.examples.tutorials.mnist import input_data

class DataSet:
    def __init__(self,mnist=False):
        self.labels = None
        self.images = None

    def load_mnist_images(self, path):
        self.images = np.load(path, allow_pickle=True)

    def load_mnist_labels(self, path):
        self.labels = np.load(path, allow_pickle=True)

    def batch(self, n):
        data = [self.images[i:i + n] for i in range(0, len(self.images), n)]
        label = [self.labels[i:i + n] for i in range(0, len(self.labels), n)]
        return data, label

    def sample(self, size):
        idx = [random.randint(0, len(self.images) - 1) for _ in range(size)]
        labels = self.labels[idx]
        images = self.images[idx]
        return images,labels



    # def loadStandandMnistDataSet(self):  # 加载测试数据集
    #     self.Mnistdataset = input_data.read_data_sets("../data/mnist_test", one_hot=True)
    #

    # def genMnistDataSetForClients(self, path, size=config.TrainDatasetSizePerClient):
    #
    #     images = self.Mnistdataset.train.images
    #     labels = self.Mnistdataset.train.labels
    #     for i in range(config.ClientNum):
    #         images2save = images[i*size:(i+1)*size]
    #         labels2save = labels[i*size:(i+1)*size]
    #         imageFilelName = path + "train_"+str(i)+"_data.npy"
    #         labelFileName = path + "train_"+str(i)+"_label.npy"
    #         with open(imageFilelName, "wb") as file:
    #             pickle.dump(images2save, file)
    #         with open(labelFileName, "wb") as file:
    #             pickle.dump(labels2save, file)
    #
    # def myCount(self,n):
    #     count = [0]*10
    #     for i in range(n):
    #         count[self.Mnistdataset.train.labels[i].argmax()] +=1
    #     return count

    @classmethod
    def merge_mnist_dataset(cls, dir, savaPath, non_iid=False):
        images = []
        labels = []
        for i in range(3):
            if non_iid:
                images.append(np.load(dir + "train_" + str(i) + "_data_nonid.npy", allow_pickle=True))
                labels.append(np.load(dir + "train_" + str(i) + "_label_nonid.npy", allow_pickle=True))
            else:
                images.append(np.load(dir + "train_" + str(i) + "_data.npy", allow_pickle=True))
                labels.append(np.load(dir + "train_" + str(i) + "_label.npy", allow_pickle=True))
        imags_merged = np.vstack(images)

        label_merged = np.vstack(labels)
        with open(savaPath + "data", "wb") as f:
            imags_merged.dump(f)
        with open(savaPath + "labels", "wb") as f:
            label_merged.dump(f)


# 生成训练集

# if __name__ == "__main__":
#     # DataSet.generate_dataset("data/train_cifar/data_1", "./data/train_1.npy", 8000)
#     # DataSet.generate_dataset("data/train_cifar/test_data", "./data/mnist_test.npy", 300)
#
#     # DataSet.merge_mnist_dataset("../data/train_mnist/", "./data/train_mnist/train_MnistMerged_")
#     dataset = DataSet()
#     dataset.loadStandandMnistDataSet()
#     path = "../data/mnist_client_dataset/"
#     # for i in range(10):
#     #         count = dataset.myCount((i+1)*500)
#     dataset.genMnistDataSetForClients(path)