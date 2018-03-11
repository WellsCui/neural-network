from abc import ABCMeta, abstractmethod


class IEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def get_state(self):
        pass

    def update(self, action):
        pass

    def clone(self):
        pass

    def reverse(self):
        pass


class IAgent(metaclass=ABCMeta):
    @abstractmethod
    def act(self, environment: IEnvironment):
        pass

    @abstractmethod
    def learn_from_experience(self, experience, learn_from_winner=False):
        pass

    @abstractmethod
    def save_model(self, model_dir):
        pass



class IEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, environment: IEnvironment):
        pass


class IActionEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, environment: IEnvironment, action):
        pass


class IPolicy(metaclass=ABCMeta):
    @abstractmethod
    def suggest(self, state, side, suggest_count):
        pass


class IModel(metaclass=ABCMeta):
    @abstractmethod
    def train(self, data):
        pass
    @abstractmethod
    def predict(self, data):
        pass


class ISynchronizable(metaclass=ABCMeta):
    @abstractmethod
    def synchronize(self, action):
        pass