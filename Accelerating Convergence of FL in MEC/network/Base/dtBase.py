import abc
import threading


class DTBase:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_weights(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_weights(self, array_):
        raise NotImplementedError()

    @abc.abstractmethod
    def send_dataset(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def start_train(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def start_test(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def list_configs(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_configs(self, config):
        raise NotImplementedError()

    @abc.abstractmethod
    def check(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare_train(self):
        raise NotImplementedError()

