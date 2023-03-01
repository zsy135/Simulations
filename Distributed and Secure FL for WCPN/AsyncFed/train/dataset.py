import pickle
import random
import numpy as np

import config


class DataSet:
    def __init__(self, path=None, mnist=False):
        if mnist:
            self.labels = None
            self.images = None
        else:
            # 处理cifar数据集
            if path is None:
                self.labels = None
                self.images = None
                self.data_set = None
            else:
                f = open(path, "rb")
                self.data_set = pickle.load(f, encoding='bytes')
                f.close()
                self.format_dataset()

    def load_mnist_images(self, path):

        self.images = np.load(path, allow_pickle=True)

    def load_mnist_labels(self, path):

        self.labels = np.load(path, allow_pickle=True)

    def format_dataset(self):
        labels_ = np.array(self.data_set[b'labels'])
        labels = []
        for i in labels_:
            tmp = [0.0] * 10
            tmp[i] = 1.0
            labels.append(tmp)
        self.labels = np.array(labels)
        images = self.data_set[b'data']
        tmp = []
        for i in range(len(images)):
            image: np.ndarray = images[i, :]
            tmp.append(image.reshape((3, 32, 32)).transpose(1, 2, 0))
        self.images = np.array(tmp)

    def batch(self, n):
        data = [self.images[i:i + n] for i in range(0, len(self.images), n)]
        label = [self.labels[i:i + n] for i in range(0, len(self.labels), n)]
        return data, label

    def sample(self, size):
        idx = [random.randint(0, len(self.images) - 1) for _ in range(size)]
        labels = self.labels[idx]
        images = self.images[idx]
        return labels, images

    @classmethod
    def generate_dataset(cls, path, savaPath, size=config.TrainDatasetSizePerClient):  # cifar
        f = open(path, "rb")
        data_set = pickle.load(f, encoding='bytes')
        f.close()
        labels = np.array(data_set[b'labels'])
        images = np.array(data_set[b'data'])
        count_dict = {}

        for i in range(size):
            if count_dict.get(labels[i]) is None:
                count_dict[labels[i]] = [i]
            else:
                count_dict[labels[i]].append(i)
        # imageNumPerClass = config.TrainDatasetSizePerClient // 10
        # idxs = []
        # for idx in range(imageNumPerClass):
        #     for i in range(10):
        #         pos_s = count_dict[i]
        #         idxs.append(pos_s[idx])
        images2save = images[:size]
        labels2save = labels[:size]
        save_dict = {b"data": images2save, b"labels": labels2save}
        with open(savaPath, "wb") as file:
            pickle.dump(save_dict, file)

    @classmethod
    def generate_dataset_mnist(cls, size=800, number=5, iid=True):  # mnist
        with open("../data/minst/mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        labels_ = mnist["training_labels"]
        images = mnist["training_images"]
        labels = []
        for i in labels_:
            tmp = [0.0] * 10
            tmp[i] = 1.0
            labels.append(tmp)
        labels = np.array(labels)
        if iid:
            return cls.generate_dataset_mnist_iid(number, images, labels, size)
        else:
            return cls.generate_dataset_mnist_noiid(number, images, labels, size)

    @classmethod
    def generate_malicious_dataset_mnist(cls, size=config.TrainDatasetSizePerClient, number=5, iid=True):  # mnist
        with open("../data/minst/mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        labels_ = mnist["training_labels"]
        images = mnist["training_images"]
        labels = []
        for i in labels_:
            tmp = [0.0] * 10
            tmp[i] = 1.0
            labels.append(tmp)
        labels = np.array(labels)

        return cls.doMalicious_dataset_mnist(number, images, labels, size)

    @classmethod
    def doMalicious_dataset_mnist(cls, number, images, labels, size):  # mnist
        images_list = []
        labels_list = []
        # 6 vs 2, 8 vs 4
        slabel = [0.0] * 10
        slabel[2] = 1.0
        tlabel = [0.0] * 10
        tlabel[6] = 1.0
        for i in range(number):
            idx = [random.randint(0, len(images) - 1) for _ in range(size)]
            labels_ = labels[idx]
            idxs = [i for i in range(len(idx)) if np.all(labels[idx[i]] == slabel)]
            idxt = [i for i in range(len(idx)) if np.all(labels[idx[i]] == tlabel)]
            labels_[idxt] = slabel
            labels_[idxs] = tlabel
            images_list.append(images[idx])
            labels_list.append(labels_)
        return np.array(images_list), np.array(labels_list)

    @classmethod
    def generate_dataset_mnist_iid(cls, number, images, labels, size):  # mnist
        images_list = []
        labels_list = []
        for i in range(number):
            idx = [random.randint(0, len(images) - 1) for _ in range(size)]
            images_list.append(images[idx])
            labels_list.append(labels[idx])
        return np.array(images_list), np.array(labels_list)

    @classmethod
    def generate_dataset_mnist_noiid(cls, number, images, labels, size):  # mnist
        # number >=3

        images_list = []
        labels_list = []
        group_labels = [[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 0, 1, 2]]
        for i in range(number):
            size_need = size
            idx = []
            group_idx = i % len(group_labels)
            while size_need > 0:
                t = [random.randint(0, len(images) - 1) for _ in range(size_need)]
                t = [v_i for v_i in t if np.argmax(labels[v_i]) in group_labels[group_idx]]
                idx.extend(t)
                size_need -= len(t)

            images_list.append(images[idx])
            labels_list.append(labels[idx])
        return np.array(images_list), np.array(labels_list)

    def save(self, name):
        file = open(name + "_label.npy", "wb")
        self.labels.dump(file)
        file.close()
        file = open(name + "_data.npy", "wb")
        self.images.dump(file)
        file.close()

    def load(self, path):
        file = path + "_label.npy"
        self.labels = np.load(file, allow_pickle=True)
        file = path + "_label.npy"
        self.images = np.load(file, allow_pickle=True)

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

if __name__ == "__main__":
    # DataSet.generate_dataset("data/train_cifar/data_1", "./data/train_1.npy", 8000)
    # DataSet.generate_dataset("data/train_cifar/test_data", "./data/mnist_test.npy", 300)

    # DataSet.merge_mnist_dataset("../data/train_mnist/", "./data/train_mnist/train_MnistMerged_")

    # 为联邦学习中的每个客户端产生新的数据集
    client_num = 2

    # images, labels = DataSet.generate_dataset_mnist(number=client_num)
    # images_noniid, labels_noniid = DataSet.generate_dataset_mnist(number=client_num, iid=False)
    # path = "../data/mnist_client_dataset/"
    # name_data = "train_{}_data.npy"
    # name_label = "train_{}_label.npy"
    # name_data_noniid = "train_{}_data_nonid.npy"
    # name_label_noniid = "train_{}_label_nonid.npy"
    # for i in range(client_num):
    #     images[i, :, :].dump(path+name_data.format(i))
    #     labels[i, :, :].dump(path+name_label.format(i))
    #     images_noniid[i, :, :].dump(path+name_data_noniid.format(i))
    #     labels_noniid[i, :, :].dump(path+name_label_noniid.format(i))


# gen mailious dataset

    # images, labels = DataSet.generate_malicious_dataset_mnist(number=client_num)
    # # images_noniid, labels_noniid = DataSet.generate_dataset_mnist(number=client_num, iid=False)
    # path = "../data/mnist_client_dataset/"
    # name_data = "train_{}_malicious_data.npy"
    # name_label = "train_{}_malicious_label.npy"
    # # name_data_noniid = "train_{}_data_nonid.npy"
    # # name_label_noniid = "train_{}_label_nonid.npy"
    # for i in range(client_num):
    #     images[i, :, :].dump(path + name_data.format(i))
    #     labels[i, :, :].dump(path + name_label.format(i))
    #     # images_noniid[i, :, :].dump(path+name_data_noniid.format(i))
    #     # labels_noniid[i, :, :].dump(path+name_label_noniid.format(i))

# gen test dataset
    images, labels = DataSet.generate_dataset_mnist(number=1,size=config.TestDatasetSize)

    path = "../data/mnist_client_dataset/"
    name_data = "test_data.npy"
    name_label = "test_label.npy"
    for i in range(1):
        images[i, :, :].dump(path+name_data.format(i))
        labels[i, :, :].dump(path+name_label.format(i))

    # print(images, labels)
