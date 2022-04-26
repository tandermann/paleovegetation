import multiprocessing.pool
import time


def waste_time(duration):
    time.sleep(duration)
    return 'Done.'


cpus = 5
pool = multiprocessing.Pool(cpus)
args = [60]*cpus
out_info = pool.map(waste_time, args)
pool.close()