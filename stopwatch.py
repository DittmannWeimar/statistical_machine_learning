import time

__name__ = "stopwatch"

class Stopwatch:
    start_time = 0
    end_time = 0

    def __init__(self, start_now = False):
        if (start_now):
            self.start()

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        return self.elapsed()

    def reset(self):
        time_elapsed = self.elapsed();
        self.end_time = 0
        self.start_time = time.time();
        return time_elapsed;

    def end_and_print(self):
        self.end()
        self.print()
        return self.elapsed()

    def end_and_printf(self):
        self.end()
        self.printf()
        return self.elapsed()

    def elapsed(self):
        return self.end_time - self.start_time

    def print(self):
        print(self.elapsed())

    def printf(self):
        sec = self.elapsed()
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))