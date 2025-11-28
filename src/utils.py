import time
import os
import numpy as np


# Print wrapper for printing the current time
def tprint(*args, pre = '', **kwargs):
    ptn = '\033[{}m'.format
    t = time.strftime(f'{ptn(96)}[%H:%M:%S] ')
    print(str(pre) + t, end=ptn(0))
    print(*args, **kwargs)
    return time.time()


# Returns the time elapsed since a given Unix time value
def tsince(t, abbrev=True, sep=' '):
    t = int(time.time() - t)
    units = ['second', 'minute', 'hour', 'day', 'year']
    output, per = [], [60, 60, 24, 365, None]
    for u, div in zip(units, per):
        e = t if div is None else t % div
        u = u[0] if abbrev else f' {u}{"s"[:e > 1]}'
        output.append(str(e) + u)
        if div is not None: t //= div
        if t < 1: break
    return sep.join(output[::-1])


# Returns the size of a given file/directory
def getsize(path, prec=2, string=True):
    size = 0
    if os.path.isfile(path):
        try: size += os.path.getsize(path)
        except: pass
    if os.path.isdir(path):
        try: subpaths = os.listdir(path)
        except: subpaths = []
        for sub in subpaths:
            sub = os.path.join(path, sub)
            size += getsize(sub, prec + 1, False)
    if not string: return size
    i = min(np.log10(size) // 3, 4) if size else 0
    size = round(size / (1000 ** i), prec)
    return f'{size} {["", *"KMGT"][int(i)]}B'


def average_dictionary(dicts):
    avg_dict = {k:0 for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            avg_dict[k] += v / len(dicts)
    return avg_dict