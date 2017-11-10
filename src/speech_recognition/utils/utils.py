import os
import time
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """Write log immediately to the disk"""
    def __init__(self, filepath):
        self.f = open(filepath, 'w')
        self.fid = self.f.fileno()
        self.filepath = filepath

    def close(self):
        self.f.close()

    def write(self, content):
        self.f.write(content)
        self.f.flush()
        os.fsync(self.fid)

    def write_buf(self, content):
        self.f.write(content)


def to_numpy(x):
    return x.data.cpu().numpy()