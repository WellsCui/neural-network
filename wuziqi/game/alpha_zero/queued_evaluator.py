# import game.alpha_zero.alpha_zero_net as azn
import queue
import concurrent.futures as futures
import threading
import time


class QueuedEvaluator(object):
    def __init__(self, batch_evaluator, batch_size):
        self.batch_evaluator = batch_evaluator
        self.batch_size = batch_size
        self.requests = queue.Queue()
        self.process_thread = None
        self._stop_event = None
        self._idle_sleep = 3

    def submit_request(self, state, future: futures.Future):
        self.requests.put((state, future))

    def stopped(self):
        return self._stop_event is None or self._stop_event.is_set()

    def fetch_request(self):
        try:
            return self.requests.get(True, 3)
        except:
            return None

    def evaluate(self, batch):
        states = [s for s, _ in batch]
        fs = [f for _, f in batch]
        results = self.batch_evaluator.batch_evaluate(states)
        # print('batch result shape:', len(results))
        # print(results[0])
        for i in range(len(fs)):
            fs[i].set_result(results[i])

    def process_requests(self):
        while not self.stopped():
            batch = []
            request = self.fetch_request()
            while request:
                batch.append(request)
                if len(batch) == self.batch_size:
                    break
                else:
                    request = self.fetch_request()
            if len(batch) > 0:
                self.evaluate(batch)
            else:
                print('No requests in queue, wait %d seconds ...' % self._idle_sleep)
                time.sleep(self._idle_sleep)
        print('evaluator stopped.')

    def start(self):
        self.process_thread = threading.Thread(target=self.process_requests)
        self._stop_event = threading.Event()
        self.process_thread.start()
        print('evaluator started.')

    def stop(self):
        self._stop_event.set()


def queued_valuator_test():
    class MockedBatchEvaluator(object):
        def batch_evaluate(self, states):
            time.sleep(1)
            return ['eval of %s' % s for s in states]
    valuator = QueuedEvaluator(MockedBatchEvaluator(), 5)
    valuator.start()
    time.sleep(5)
    requests = [(i, futures.Future()) for i in range(100)]
    for state, future in requests:
        valuator.submit_request(state, future)

    print('getting result...')

    for state, future in requests:
        print('result of state %s: %s' % (state, future.result()))
    valuator.stop()


# queued_valuator_test()