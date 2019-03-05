from abc import abstractmethod


class Trainer:
    @abstractmethod
    def train(self):
        raise NotImplementedError