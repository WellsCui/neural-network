
import game.interfaces as interfaces
import threading


class SynchronizableObject(interfaces.ISynchronizable):
    def __init__(self):
        self.locker = threading.Lock()

    def synchronize(self, action):
        self.locker.acquire()
        try:
            action()
        finally:
            self.locker.release()