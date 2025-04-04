import sys

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
        print("Tee init ok")

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        print("Tee del ok")

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

