import time


# Print wrapper for printing the current time
def tprint(*args, pre = '', **kwargs):
    ptn = '\033[{}m'.format
    t = time.strftime(f'{ptn(96)}[%H:%M:%S] ')
    print(str(pre) + t, end=ptn(0))
    print(*args, **kwargs)
    return time.time()