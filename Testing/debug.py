import sys
from Testing.config import DEBUG


class DebugOutput:

    def __init__(self):

        self.original_stdout = sys.stdout

    def write(self, text):

        if DEBUG:
            self.original_stdout.write(text)

    def flush(self):

        self.original_stdout.flush()


def setup_debug():

    sys.stdout = DebugOutput()