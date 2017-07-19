from abc import ABCMeta, abstractmethod


class IEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def get_state(self):
        pass

    def update(self, action):
        pass


class IAgent(metaclass=ABCMeta):
    @abstractmethod
    def act(self, environment: IEnvironment):
        pass


class IEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, environment: IEnvironment):
        pass


class IPolicy(metaclass=ABCMeta):
    @abstractmethod
    def resolve(self, environment: IEnvironment):
        pass

class ITrainable(metaclass=ABCMeta):
    @abstractmethod
    def train(self, data):
        pass