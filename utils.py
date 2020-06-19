
'''
Copyright (C) 2020  Daniel Duque (see license.txt)
'''
import numpy as np
import time


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def roundup(x, l):
    return int(np.ceil(x / l)) * l


def round_closest(x, l):
    return int(np.around(x / l)) * l


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw and kw['log_time'] is not None:
            name = kw.get('log_name', method.__name__.upper())
            call_time = (te - ts)
            if name not in kw['log_time']:
                kw['log_time'][name] = {'calls': 1, 'max_time': call_time, 'avg_time': call_time}
            else:
                avg = kw['log_time'][name]['avg_time']
                calls = kw['log_time'][name]['calls']
                kw['log_time'][name]['avg_time'] = (avg * calls + call_time) / (calls + 1)
                kw['log_time'][name]['calls'] = calls + 1
                kw['log_time'][name]['max_time'] = np.maximum(kw['log_time'][name]['max_time'], call_time)
        
        else:
            print(f'{method.__name__}: {(te - ts): 10.5f} s')
        return result
    
    return timed