from multiprocessing import Pool, Queue, cpu_count
from threading import RLock
from typing import Callable

class PQueue:
    def __init__(task: Callable):
        self._lock = RLock()
        self._active = 0
        self._done = False
        self._q_task = Queue()
        self._q_result = Queue()
        self._pool = None
        self._task = task

    def push(item):
        with self._lock:
            self._q_task.push(item)

    def generate():
        self._q_result.clear()
        self._done = False
        self._active = 0

        cpu = cpu_count()
        self._pool = Pool(cpu)

        def run_tasks():
            while not self._done:
                t = None

                with self._lock:
                    try:
                        t = self._q_task.get_nowait()
                    except Queue.Empty:
                        t = None

                    if t is not None:
                        self._active += 1

                if t is not None:
                    for r in self._task(t):
                        self._q_result.put(r, 100)

                    with self._lock:
                        self._active -= 1

        for _ in range(cpu):
            self._pool.apply_async(run_tasks, ())

        while not self._done:
            with self._lock:
                if not self._active and self._q_task.empty():
                    self._done = True

            while not self._q_result.empty():
                r = self._q_result.get(True, 100)
                if r is not None:
                    yield r

        self._pool.join()
        self._pool.close()
        self._pool = None
